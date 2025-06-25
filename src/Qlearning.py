from src.memory import ReplayMemory
from src.visualization import Visualization
from src.normalizer import Normalizer
from src.desra import DESRA
from src.sumo import SUMO
from src.accident_manager import AccidentManager

import traci
import numpy as np
import random
import time

GREEN_ACTION = 0
RED_ACTION = 1

def index_to_action(index, actions_map):
    return actions_map[index]["phase"], actions_map[index]["duration"]

def phase_to_index(phase, actions_map, duration):
    for i, action in actions_map.items():
        if action["phase"] == phase:
            return i

class Simulation(SUMO):
    def __init__(
        self,
        memory: ReplayMemory,
        visualization: Visualization,
        agent_cfg,
        max_steps,
        traffic_lights,
        accident_manager: AccidentManager,
        interphase_duration=3,
        epoch=1000,
        path=None,
    ):
        self.memory = memory
        self.visualization = visualization
        self.agent_cfg = agent_cfg
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.interphase_duration = interphase_duration
        self.epoch = epoch
        self.path = path
        self.weight = agent_cfg["weight"]
        self.green_time_normalizer = Normalizer()
        self.outflow_rate_normalizer = Normalizer()
        self.density_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_speed_normalizer = Normalizer()

        self.step = 0
        self.num_actions = {}
        self.actions_map = {}

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

        self.history = {
            "agent_reward": {},
            "travel_speed": {},
            "travel_time": {},
            "density": {},
            "outflow_rate": {},
            "green_time": {},
            "q_value": {},
            "max_next_q_value": {},
            "target": {},
            "loss": {},
        }

        self.initState()

        # Q-learning parameters
        self.q_table = {}  # {(state_tuple, action): value}
        self.alpha = 0.1   # learning rate
        self.gamma = self.agent_cfg["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = self.agent_cfg.get("min_epsilon", 0.01)
        self.epsilon_decay = self.agent_cfg.get("decay_rate", 0.995)

        self.desra = DESRA(interphase_duration=self.interphase_duration)
        self.longest_phase, self.longest_phase_len, self.longest_phase_id = self.get_longest_phase()

    def get_longest_phase(self):
        max_len = -1
        longest_phase = None
        longest_id = None

        for tl_id, actions in self.actions_map.items():
            for action in actions.values():
                phase = action["phase"]
                if len(phase) > max_len:
                    max_len = len(phase)
                    longest_phase = phase
                    longest_id = tl_id

        return longest_phase, max_len, longest_id

    def initState(self):
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            self.agent_memory[traffic_light_id] = self.memory

            self.green_time[traffic_light_id] = 20
            self.green_time_old[traffic_light_id] = 20
            self.num_actions[traffic_light_id] = len(
                self.agent_cfg["green_duration_deltas"]
            ) * len(traffic_light["phase"])
            self.actions_map[traffic_light_id] = {}

            i = 0
            for phase in traffic_light["phase"]:
                for green_delta in self.agent_cfg["green_duration_deltas"]:
                    self.actions_map[traffic_light_id][i] = {
                        "phase": phase,
                        "duration": green_delta,
                    }
                    i += 1

            self.agent_reward[traffic_light_id] = 0
            self.agent_state[traffic_light_id] = 0
            self.agent_old_state[traffic_light_id] = 0
            self.old_outflow_rate[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.old_travel_speed[traffic_light_id] = 0
            self.travel_speed[traffic_light_id] = 0
            self.old_travel_time[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.old_density[traffic_light_id] = 0
            self.density[traffic_light_id] = 0

            for key in self.history:
                self.history[key][traffic_light_id] = []

    def get_output_dims(self):
        output_dims = []
        for traffic_light in self.traffic_lights:
            output_dims.append(self.num_actions[traffic_light["id"]])
        return output_dims

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def set_q(self, state, action, value):
        self.q_table[(tuple(state), action)] = value

    def select_action(self, traffic_light_id, state, epsilon):
        # Epsilon-greedy for tabular Q-learning
        if random.random() < epsilon:
            return random.randint(0, self.num_actions[traffic_light_id] - 1)
        else:
            q_values = [self.get_q(state, a) for a in range(self.num_actions[traffic_light_id])]
            return int(np.argmax(q_values))

    def qlearn_update(self, state, action, reward, next_state, done, traffic_light_id):
        max_next_q = max([self.get_q(next_state, a) for a in range(self.num_actions[traffic_light_id])])
        old_q = self.get_q(state, action)
        target = reward + (0 if done else self.gamma * max_next_q)
        new_q = old_q + self.alpha * (target - old_q)
        self.set_q(state, action, new_q)

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start = time.time()

        while self.step < self.max_steps:
            for traffic_light in self.traffic_lights:
                traffic_light_id = traffic_light["id"]
                state, green_time = self.get_state(traffic_light)
                action_idx = self.select_action(
                    traffic_light_id,
                    state,
                    epsilon,
                )

                phase, green_delta = index_to_action(
                    action_idx,
                    self.actions_map[traffic_light_id],
                )
                green_time = max(green_time, self.green_time[traffic_light_id] + green_delta)

                self.green_time_old[traffic_light_id] = self.green_time[traffic_light_id]
                self.green_time[traffic_light_id] = green_time

                self.old_outflow_rate[traffic_light_id] = self.outflow_rate[traffic_light_id]
                self.old_travel_speed[traffic_light_id] = self.travel_speed[traffic_light_id]
                self.old_travel_time[traffic_light_id] = self.travel_time[traffic_light_id]
                self.old_density[traffic_light_id] = self.density[traffic_light_id]

                old_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase)

                self.set_green_phase(
                    traffic_light_id,
                    green_time,
                    phase,
                )

                travel_speed = 0
                density = 0

                green_time = min(green_time, self.max_steps - self.step)

                for _ in range(green_time):
                    self.accident_manager.create_accident(current_step=self.step) 
                    self.step += 1
                    traci.simulationStep()

                    new_vehicle_ids = self.get_vehicles_in_phase(traffic_light, phase)
                    outflow = sum(
                        1 for vid in old_vehicle_ids if vid not in new_vehicle_ids
                    )

                    travel_speed += self.get_avg_speed(traffic_light)
                    density += self.get_avg_density(traffic_light)

                    old_vehicle_ids = new_vehicle_ids

                self.outflow_rate[traffic_light_id] = (
                    outflow / green_time if green_time > 0 else 0
                )

                self.travel_speed[traffic_light_id] = (
                    travel_speed / green_time if green_time > 0 else 0
                )
                self.travel_time[traffic_light_id] = (
                    self.get_avg_travel_time(traffic_light) / green_time
                    if green_time > 0
                    else 0
                )
                self.density[traffic_light_id] = (
                    density / green_time if green_time > 0 else 0
                )

                reward = self.get_reward(traffic_light_id)
                done = self.step >= self.max_steps

                next_state, _ = self.get_state(traffic_light)
                self.agent_memory[traffic_light_id].push(
                    state, action_idx, reward, next_state, done
                )

                self.qlearn_update(state, action_idx, reward, next_state, done, traffic_light_id)

                self.history["agent_reward"][traffic_light_id].append(reward)
                self.history["travel_speed"][traffic_light_id].append(
                    self.travel_speed[traffic_light_id]
                )
                self.history["travel_time"][traffic_light_id].append(
                    self.travel_time[traffic_light_id]
                )
                self.history["density"][traffic_light_id].append(
                    self.density[traffic_light_id]
                )
                self.history["outflow_rate"][traffic_light_id].append(
                    self.outflow_rate[traffic_light_id]
                )
                self.history["green_time"][traffic_light_id].append(
                    self.green_time[traffic_light_id]
                )

        simulation_time = time.time() - start

        traci.close()

        print("Simulation ended")
        print("---------------------------------------")

        self.save_plot(episode=episode)

        self.step = 0

        total_reward = 0
        for traffic_light in self.traffic_lights:
            total_reward += np.sum(self.history["agent_reward"][traffic_light["id"]])

        print("Total reward:", total_reward, "- Epsilon:", epsilon)

        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return simulation_time, 0

    def get_reward(self, traffic_light_id):
        return (
            self.weight["green_time"]
            * self.green_time_normalizer.normalize(
                self.green_time_old[traffic_light_id]
                - self.green_time[traffic_light_id]
            )
            + self.weight["outflow_rate"]
            * self.outflow_rate_normalizer.normalize(
                self.outflow_rate[traffic_light_id]
                - self.old_outflow_rate[traffic_light_id]
            )
            + self.weight["travel_speed"]
            * self.travel_speed_normalizer.normalize(
                self.travel_speed[traffic_light_id]
                - self.old_travel_speed[traffic_light_id]
            )
            + self.weight["travel_time"]
            * self.travel_time_normalizer.normalize(
                self.old_travel_time[traffic_light_id]
                - self.travel_time[traffic_light_id]
            )
            + self.weight["density"]
            * self.density_normalizer.normalize(
                self.old_density[traffic_light_id] - self.density[traffic_light_id]
            )
        )

    def save_plot(self, episode):
        avg_history = {}

        for metric, data_per_tls in self.history.items():
            data_lists = [data for data in data_per_tls.values() if len(data) > 0]
            if not data_lists:
                continue

            min_length = min(len(data) for data in data_lists)
            data_lists = [data[:min_length] for data in data_lists]

            avg_data = [
                sum(step_vals) / len(step_vals) for step_vals in zip(*data_lists)
            ]

            avg_history[metric] = avg_data[-100:]

        if episode % 100 == 0:
            print(f"Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data_and_plot(
                    data=data,
                    filename=f"{metric}_avg_episode_{episode}",
                    xlabel="Step",
                    ylabel=metric.replace("_", " ").title(),
                )
            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def set_yellow_phase(self, phase):
        traci.trafficlight.setPhase(self.traffic_light_id, phase)
        traci.trafficlight.setPhaseDuration(
            self.traffic_light_id, self.interphase_duration
        )

    def set_green_phase(self, tlsId, duration, new_phase):
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)

    def update_phase(self, phase_idxs, num_phases):
        phase_str = num_phases * ["r"]
        return self.replace_chars(phase_str, phase_idxs, "G")

    def replace_chars(self, s, indexes, replacements):
        s_list = list(s)

        if isinstance(replacements, str):
            for i in indexes:
                if 0 <= i < len(s_list):
                    s_list[i] = replacements
        else:
            for i, c in zip(indexes, replacements):
                if 0 <= i < len(s_list):
                    s_list[i] = c

        return "".join(s_list)

    def get_state(self, traffic_light):
        state = [
            0,
            0,
            0,
            0,
            [],
            0,
        ]  # min_free_capacity, density, waiting_time, queue_length, phase_one_hot, green_time

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

        phase, green_time = self.desra.select_phase(traffic_light)
        state[4] = phase_to_index(phase, self.actions_map[traffic_light["id"]], 0)
        state[5] = green_time

        return np.array(state, dtype=np.float32), green_time

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
        occupied_length = traci.lanearea.getLastStepOccupancy(detector_id) / 100 * lane_length

        return lane_length - occupied_length

    def get_density(self, detector_id):
        """
        Get the density of vehicles on a lane.
        """
        return traci.lanearea.getLastStepOccupancy(detector_id) / 100

    def get_waiting_time(self, detector_id):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

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
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
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
