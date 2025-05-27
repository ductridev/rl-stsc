from src.model import DQN
from src.memory import ReplayMemory

import traci
import numpy as np
import random
import torch

GREEN_ACTION = 0
RED_ACTION = 1

def index_to_action(index, green_duration_deltas):
    phase_idx = index // len(green_duration_deltas)
    delta_idx = index % len(green_duration_deltas)
    return phase_idx, green_duration_deltas[delta_idx]

def action_to_index(phase_idx, delta_idx, num_deltas):
    return phase_idx * num_deltas + delta_idx

class Simulation:
    def __init__(
        self,
        agent: DQN,
        agent_memory: ReplayMemory,
        agent_cfg,
        max_steps,
        traffic_lights,
        interphase_duration=3,
        epoch=1000,
    ):
        self.agent = agent
        self.agent_memory = agent_memory
        self.agent_cfg = agent_cfg
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.interphase_duration = interphase_duration
        self.epoch = epoch

        self.agent_action = {}
        self.agent_old_action = {}
        self.agent_reward = 0
        self.agent_state = {}
        self.agent_old_state = {}

        self.outflow_rate = 0
        self.old_outflow_rate = 0
        self.travel_speed = 0
        self.travel_time = 0
        self.density = 0
        self.old_travel_speed = 0
        self.old_travel_time = 0
        self.old_density = 0
        self.step = 0
        self.green_time = 0
        self.green_time_old = 0

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        while self.step < self.max_steps:
            for traffic_light in self.traffic_lights:
                state = self.get_state(traffic_light)
                action_idx = self.select_action(state, epsilon)

                num_deltas = len(self.agent_cfg["green_duration_deltas"])
                phase_idx, green_delta = index_to_action(action_idx, num_deltas, self.get_num_phases(traffic_light))
                green_time = max(1, 10 + green_delta)

                self.green_time_old = self.green_time
                self.green_time = green_time

                self.old_outflow_rate = self.outflow_rate
                self.old_travel_speed = self.travel_speed
                self.old_travel_time = self.travel_time
                self.old_density = self.density

                old_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase_idx)

                self.set_green_phase(
                    traffic_light["id"],
                    green_time,
                    [phase_idx],
                    self.get_num_phases(traffic_light),
                    traffic_light
                )

                for _ in range(min(green_time, self.max_steps - self.step)):
                    self.step += 1
                    traci.simulationStep()

                new_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase_idx)
                outflow = sum(1 for vid in old_vehicle_ids if vid not in new_vehicle_ids)
                self.outflow_rate = outflow / green_time if green_time > 0 else 0

                # Get new traffic metrics
                self.travel_speed = self.get_avg_speed(traffic_light)
                self.travel_time = self.get_avg_travel_time(traffic_light)
                self.density = self.get_avg_density(traffic_light)

                reward = self.get_reward()
                print(f"Phase: {phase_idx}, Green time: {green_time}, Reward: {reward}")

                next_state = self.get_state(traffic_light)
                self.agent_memory.push(state, action_idx, reward, next_state)

                self.agent.train(self.agent_memory)
    
    def train_agent(self, traffic_light, epsilon):
        state = {}
        self.agent_old_action = self.agent_action
        self.agent_old_state = self.agent_state

        for detector in traffic_light["detectors"]:
            queue_length = self.get_queue_length(detector["id"])
            if detector["phase"] not in state:
                state[detector["phase"]] = 0
            state[detector["phase"]] += queue_length

        self.agent_state = state

        state_tensor = torch.from_numpy(np.array([list(state.values())])).float()

        if random.random() < epsilon:
            phase_idx = random.randint(0, len(traffic_light["phases"]) - 1)
            delta = random.choice(self.agent_cfg["green_duration_deltas"])
        else:
            q_values: torch.Tensor = self.agent.predict(state_tensor)
            action_idx = torch.argmax(q_values).item()
            phase_idx, delta = index_to_action(
                action_idx,
                self.agent_cfg["green_duration_deltas"],
                len(traffic_light["phases"]),
            )

        self.agent_action[traffic_light["id"]] = {
            "phase": phase_idx,
            "duration": max(1, 10 + delta)
        }

    def get_reward(self):
        return (
            self.agent_reward
            + (self.green_time_old - self.green_time)
            + (self.outflow_rate - self.old_outflow_rate)
            + (self.travel_speed - self.old_travel_speed)
            + (self.travel_time - self.old_travel_time)
            + (self.density - self.old_density)
        )


    def set_yellow_phase(self, phase):
        """
        Set the traffic light to yellow phase.
        """
        traci.trafficlight.setPhase(self.traffic_light_id, phase)
        traci.trafficlight.setPhaseDuration(
            self.traffic_light_id, self.interphase_duration
        )

    def set_green_phase(self, tlsId, duration, phase_idxs, traffic_light):
        new_phase = traffic_light["phase"][phase_idxs[0]]  # Use custom-defined phase string
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)


    def update_phase(self, phase_idxs, num_phases):
        """
        Update the phase of the traffic light
        """
        phase_str = num_phases * ["r"]
        return self.replace_chars(phase_str, phase_idxs, "G")

    def replace_chars(self, s, indexes, replacements):
        s_list = list(s)

        if isinstance(replacements, str):
            # One replacement character for all indices
            for i in indexes:
                if 0 <= i < len(s_list):
                    s_list[i] = replacements
        else:
            # Each index has a corresponding replacement
            for i, c in zip(indexes, replacements):
                if 0 <= i < len(s_list):
                    s_list[i] = c

        return "".join(s_list)

    def select_action(self, state, epsilon):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state (np.array): current state of the simulation
            epsilon (float): exploration rate for epsilon-greedy policy
        Returns:
            int: selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.agent_cfg['num_actions'] - 1)
        else:
            state = torch.from_numpy(np.array([state])).float()
            with torch.no_grad():
                q_values: torch.Tensor = self.agent.predict(state)
                return q_values.argmax().item()

    def get_state(self, detector_id):
        """
        Get the current state of the simulation.
        Returns:
            np.array: current state of the simulation
        """
        # Get the current state from the simulation
        state = np.zeros((self.agent_cfg["num_states"],))
        min_free_capacity = self.get_min_free_capacity(detector_id)
        density = self.get_density(detector_id)
        waiting_time = self.get_waiting_time(detector_id)
        queue_length = self.get_queue_length(detector_id)
        if state == np.zeros((self.num_states,)):
            state = np.array(
                [
                    density,
                    min_free_capacity,
                    waiting_time,
                    queue_length
                ]
            )
        else:
            state = np.append(
                state,
                np.array(
                    [
                        density,
                        min_free_capacity,
                        waiting_time,
                        queue_length,
                    ]
                ),
                axis=0,
            )
        return state

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_min_free_capacity(self, detector_id):
        """
        Get the minimum free capacity of a lane.
        """
        lane_length = traci.lanearea.getLength(detector_id)
        occupied_length = traci.lanearea.getLastStepOccupancy(detector_id) * lane_length

        return lane_length - occupied_length

    def get_density(self, detector_id):
        """
        Get the density of vehicles on a lane.
        """
        return traci.lanearea.getLastStepOccupancy(detector_id)

    def get_waiting_time(self, detector_id):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = traci.lane.getLastStepVehicleIDs(detector_id)

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    def get_travel_speed(self, edge_id):
        """
        Get the travel speed of vehicles on a lane.
        """
        return traci.edge.getLastStepMeanSpeed(edge_id)

    def get_travel_time(self, edge_id):
        """
        Get the travel time of vehicles on a lane.
        """
        return traci.edge.getTraveltime(edge_id)
    def get_avg_speed(self, traffic_light):
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lane.getLastStepMeanSpeed(detector["id"])
                speeds.append(speed)
            except:
                pass
        return np.mean(speeds) if speeds else 0.0
    def get_avg_travel_time(self, traffic_light):
        travel_times = []
        for detector in traffic_light["detectors"]:
            try:
                travel_time = traci.lane.getTraveltime(detector["id"])
                travel_times.append(travel_time)
            except:
                pass
        return np.mean(travel_times) if travel_times else 0.0

    def get_avg_density(self, traffic_light):
        densities = []
        for detector in traffic_light["detectors"]:
            try:
                densities.append(traci.lanearea.getLastStepOccupancy(detector["id"]))
            except:
                pass
        return np.mean(densities) if densities else 0.0
    def get_num_phases(self, traffic_light):
        """
        Returns the number of custom-defined traffic light phases.
        """
        return len(traffic_light["phase"])
    def get_vehicles_in_phase(self, traffic_light, phase_idx):
        """
        Returns the vehicle IDs on lanes with a green signal in the specified phase.
        """
        phase_str = traffic_light["phase"][phase_idx]

        green_lanes = [
            traffic_light["controlled_lanes"][i]
            for i, light_state in enumerate(phase_str)
            if light_state.upper() == "G"
        ]

        vehicle_ids = []
        for lane in green_lanes:
            try:
                vehicle_ids.extend(traci.lane.getLastStepVehicleIDs(lane))
            except:
                pass  # Skip any invalid or unavailable lanes

        return vehicle_ids
