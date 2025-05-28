from src.model import DQN
from src.memory import ReplayMemory

import traci
import numpy as np
import random
import torch
import time
GREEN_ACTION = 0
RED_ACTION = 1


def index_to_action(index, actions_map):
    return actions_map[index]['phase'], actions_map[index]['duration']


def action_to_index(phase_idx, delta_idx, num_deltas):
    return phase_idx * num_deltas + delta_idx


class Simulation:
    def __init__(
        self,
        memory: ReplayMemory,
        agent_cfg,
        max_steps,
        traffic_lights,
        interphase_duration=3,
        epoch=1000,
    ):
        self.memory = memory
        self.agent_cfg = agent_cfg
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.interphase_duration = interphase_duration
        self.epoch = epoch

        self.step = 0
        self.num_actions = {}
        self.actions_map = {}
        self.agents : list[DQN] = {}

        self.agent_reward = {}
        self.agent_state = {}
        self.agent_old_state = {}
        self.agent_memory = {}

        self.outflow_rate = {}
        self.old_outflow_rate = {}
        self.travel_speed = {}
        self.travel_time = {}
        self.density = {}
        self.old_travel_speed = {}
        self.old_travel_time = {}
        self.old_density = {}
        self.green_time = {}
        self.green_time_old = {}
        self.initState()

    def initState(self):
        for traffic_light in self.traffic_lights:
            self.agent_memory[traffic_light["id"]] = self.memory

            # Initialize the green time
            self.green_time[traffic_light["id"]] = 20

            # Initialize the old green time
            self.green_time_old[traffic_light["id"]] = 20

            # Initialize the number of actions
            self.num_actions[traffic_light["id"]] = len(self.agent_cfg["green_duration_deltas"]) * len(
                traffic_light["phase"]
            )

            # Initialize the action map
            self.actions_map[traffic_light["id"]] = {}

            # We will convert from number of phases and green delta to phase index and green delta
            i = 0

            for phase in traffic_light["phase"]:
                for green_delta in self.agent_cfg["green_duration_deltas"]:
                    self.actions_map[traffic_light["id"]][i] = {
                        "phase": phase,
                        "duration": green_delta,
                    }

                    i += 1

            # Initialize the agent
            self.agents[traffic_light["id"]] = DQN(
                num_layers=self.agent_cfg["num_layers"],
                batch_size=self.agent_cfg["batch_size"],
                learning_rate=self.agent_cfg["learning_rate"],
                input_dim=self.agent_cfg["num_states"],
                output_dim=self.num_actions[traffic_light["id"]],
                gamma=self.agent_cfg["gamma"],
            )

            # self.agents[traffic_light["id"]].train(True)

            # Initialize the agent reward
            self.agent_reward[traffic_light["id"]] = 0

            # Initialize the agent state
            self.agent_state[traffic_light["id"]] = 0

            # Initialize the agent old state
            self.agent_old_state[traffic_light["id"]] = 0

            # Initialize the old outflow rate
            self.old_outflow_rate[traffic_light["id"]] = 0

            # Initialize the outflow rate
            self.outflow_rate[traffic_light["id"]] = 0

            # Initialize the old travel speed
            self.old_travel_speed[traffic_light["id"]] = 0

            # Initialize the travel speed
            self.travel_speed[traffic_light["id"]] = 0

            # Initialize the old travel time
            self.old_travel_time[traffic_light["id"]] = 0

            # Initialize the travel time
            self.travel_time[traffic_light["id"]] = 0

            # Initialize the old density
            self.old_density[traffic_light["id"]] = 0

            # Initialize the density
            self.density[traffic_light["id"]] = 0

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start = time.time()

        while self.step < self.max_steps:
            for traffic_light in self.traffic_lights:
                state = self.get_state(traffic_light)
                action_idx = self.select_action(traffic_light["id"], self.agents[traffic_light["id"]], state, epsilon)

                phase, green_delta = index_to_action(
                    action_idx,
                    self.actions_map[traffic_light["id"]],
                )
                green_time = max(1, self.green_time[traffic_light["id"]] + green_delta)

                self.green_time_old[traffic_light["id"]] = self.green_time[traffic_light["id"]]
                self.green_time[traffic_light["id"]] = green_time

                self.old_outflow_rate[traffic_light["id"]] = self.outflow_rate[traffic_light["id"]]
                self.old_travel_speed[traffic_light["id"]] = self.travel_speed[traffic_light["id"]]
                self.old_travel_time[traffic_light["id"]] = self.travel_time[traffic_light["id"]]
                self.old_density[traffic_light["id"]] = self.density[traffic_light["id"]]

                old_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase)

                self.set_green_phase(
                    traffic_light["id"],
                    green_time,
                    phase,
                )

                for _ in range(min(green_time, self.max_steps - self.step)):
                    self.step += 1
                    traci.simulationStep()

                new_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase)
                outflow = sum(
                    1 for vid in old_vehicle_ids if vid not in new_vehicle_ids
                )
                self.outflow_rate[traffic_light["id"]] = outflow / green_time if green_time > 0 else 0

                # Get new traffic metrics
                self.travel_speed[traffic_light["id"]] = self.get_avg_speed(traffic_light)
                self.travel_time[traffic_light["id"]] = self.get_avg_travel_time(traffic_light)
                self.density[traffic_light["id"]] = self.get_avg_density(traffic_light)

                reward = self.get_reward(traffic_light["id"])
                print(f"Traffic light: {traffic_light['id']}, Phase: {phase}, Green time: {green_time}, Reward: {reward}")

                next_state = self.get_state(traffic_light)
                self.agent_memory[traffic_light["id"]].push(state, action_idx, reward, next_state)

        simulation_time = time.time() - start

        print("Training...")
        start_time = time.time()
        for _ in range(self.epoch):
            self.training()
        training_time = time.time() - start_time

        return simulation_time, training_time

    def training(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        for traffic_light in self.traffic_lights:
            batch = self.agent_memory[traffic_light["id"]].get_samples(self.agents[traffic_light["id"]].batch_size)

            if len(batch) > 0: 
                for (state, action, reward, next_state, done) in batch:
                    self.agents[traffic_light["id"]].update(state, action, reward, next_state, done)

    def get_reward(self, traffic_light_id):
        return (
            self.agent_reward[traffic_light_id]
            + (self.green_time_old[traffic_light_id] - self.green_time[traffic_light_id])
            + (self.outflow_rate[traffic_light_id] - self.old_outflow_rate[traffic_light_id])
            + (self.travel_speed[traffic_light_id] - self.old_travel_speed[traffic_light_id])
            + (self.travel_time[traffic_light_id] - self.old_travel_time[traffic_light_id])
            + (self.density[traffic_light_id] - self.old_density[traffic_light_id])
        )

    def set_yellow_phase(self, phase):
        """
        Set the traffic light to yellow phase.
        """
        traci.trafficlight.setPhase(self.traffic_light_id, phase)
        traci.trafficlight.setPhaseDuration(
            self.traffic_light_id, self.interphase_duration
        )

    def set_green_phase(self, tlsId, duration, new_phase):
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

    def select_action(self, traffic_light_id, agent: DQN, state, epsilon):
        """
        Select an action using epsilon-greedy policy.
        Args:
            state (np.array): current state of the simulation
            epsilon (float): exploration rate for epsilon-greedy policy
        Returns:
            int: selected action
        """
        if random.random() < epsilon:
            return random.randint(0, self.num_actions[traffic_light_id])
        else:
            state = torch.from_numpy(state).float()
            with torch.no_grad():
                q_values: torch.Tensor = agent.predict_one(state)
                return q_values.argmax().item()

    def get_state(self, traffic_light):
        """
        Get the current state at a specific traffic light in the simulation.

        Returns:
            np.ndarray: 2D array of shape (num_detectors, 4)
        """
        state = [0, 0, 0, 0]  # min_free_capacity, density, waiting_time, queue_length

        for detector in traffic_light["detectors"]:
            detector_id = detector["id"]
            min_free_capacity = self.get_min_free_capacity(detector_id)
            density = self.get_density(detector_id)
            waiting_time = self.get_waiting_time(detector_id)
            queue_length = self.get_queue_length(detector_id)

            state[0] += min_free_capacity
            state[1] += density
            state[2] += waiting_time
            state[3] += queue_length

        return np.array(state, dtype=np.float32)

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
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)

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
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                speeds.append(speed)
            except:
                pass
        return np.mean(speeds) if speeds else 0.0

    def get_avg_travel_time(self, traffic_light):
        travel_times = []
        for detector in traffic_light["detectors"]:
            try:
                travel_time = traci.lanearea.getTraveltime(detector["id"])
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

    def get_vehicles_in_phase(self, traffic_light, phase_str):
        """
        Returns the vehicle IDs on lanes with a green signal in the specified phase.
        """

        lane_idxs = []

        # Group all lane_idx in detectors
        lane_idxs = [
            lane_id
            for detector in traffic_light["detectors"]
            for lane_id in detector["lane_idx"]
        ]

        green_lanes = [
            lane_idxs[i]
            for i, light_state in enumerate(phase_str)
            if light_state.upper() == "G" and i < len(lane_idxs)
        ]

        vehicle_ids = []
        for lane in green_lanes:
            try:
                vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(lane))
            except:
                pass  # Skip any invalid or unavailable lanes

        return vehicle_ids
