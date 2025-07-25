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
import pickle
import pandas as pd

GREEN_ACTION = 0
RED_ACTION = 1


def index_to_action(index, actions_map):
    return actions_map[index]["phase"], actions_map[index]["duration"]


def phase_to_index(phase, actions_map, duration):
    for i, action in actions_map.items():
        if action["phase"] == phase:
            return i


class QSimulation(SUMO):
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
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_speed_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()
        
        # Add DQN-style normalizers
        self.green_time_normalizer = Normalizer()
        self.density_normalizer = Normalizer()

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
        self.waiting_time = {}
        self.density = {}
        self.old_travel_speed = {}
        self.old_travel_time = {}
        self.old_density = {}
        self.phase = {}

        self.history = {
            "agent_reward": {},
            "travel_speed": {},
            "travel_time": {},
            "density": {},
            "outflow": {},
            "q_value": {},
            "max_next_q_value": {},
            "target": {},
            "loss": {},
            "queue_length": {},
            "waiting_time": {},
        }

        self.initState()

        # Q-learning parameters
        self.q_table = {}  # {(state_tuple, action): value}
        self.alpha = 0.01  # Lower learning rate for function approximation
        self.gamma = self.agent_cfg["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = self.agent_cfg.get("min_epsilon", 0.01)
        self.epsilon_decay = self.agent_cfg.get("decay_rate", 0.995)

        self.desra = DESRA(interphase_duration=self.interphase_duration)
        self.longest_phase, self.longest_phase_len, self.longest_phase_id = (
            self.get_longest_phase()
        )
        
        # Initialize RBF sampler once for consistent features
        self.rbf_sampler = None
        self.rbf_weights = {}
        self.n_components = 50  # Reduced complexity
        self.rbf_gamma = 0.1    # Lower gamma for stability

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

            # Initialize the green time
            self.green_time = {}
            self.green_time_old = {}
            self.green_time[traffic_light_id] = 20
            self.green_time_old[traffic_light_id] = 20

            # Initialize the number of actions like DQN
            self.num_actions[traffic_light_id] = len(
                self.agent_cfg.get("green_duration_deltas", [-5, 0, 5])
            ) * len(traffic_light["phase"])
            self.actions_map[traffic_light_id] = {}

            # Create action map like DQN: phase + duration combinations
            i = 0
            for phase in traffic_light["phase"]:
                for green_delta in self.agent_cfg.get("green_duration_deltas", [-5, 0, 5]):
                    self.actions_map[traffic_light_id][i] = {
                        "phase": phase,
                        "duration": green_delta,
                    }
                    i += 1

            self.agent_reward[traffic_light_id] = 0
            self.agent_state[traffic_light_id] = 0
            self.agent_old_state[traffic_light_id] = 0
            
            # Initialize metrics like DQN
            self.old_outflow_rate[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.old_travel_speed[traffic_light_id] = 0
            self.travel_speed[traffic_light_id] = 0
            self.old_travel_time[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.old_density[traffic_light_id] = 0
            self.density[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0
            self.phase[traffic_light_id] = None

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
            q_values = [
                self.get_q_rbf(state, a) for a in range(self.num_actions[traffic_light_id])
            ]
            return int(np.argmax(q_values))

    def qlearn_update(self, state, action, reward, next_state, done, traffic_light_id):
        max_next_q = max(
            [
                self.get_q_rbf(next_state, a)
                for a in range(self.num_actions[traffic_light_id])
            ]
        )
        old_q = self.get_q_rbf(state, action)
        target = reward + (0 if done else self.gamma * max_next_q)
        # Use target directly for RBF function approximation
        self.set_q_rbf(state, action, target)

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start = time.time()

        # Initialize per-light state tracking
        tl_states = {}
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            tl_states[tl_id] = {
                "green_time_remaining": 0,
                "travel_speed_sum": 0,
                "travel_time_sum": 0,
                "waiting_time": 0,
                "outflow": 0,
                "queue_length": 0,
                "density_sum": 0,  # Add density_sum for DQN compatibility
                "old_vehicle_ids": [],
                "state": None,
                "action_idx": None,
                "phase": None,
                "green_time": 20,
                "step_travel_speed_sum": 0,
                "step_travel_time_sum": 0,
                "step_density_sum": 0,
                "step_outflow": 0,
                "step_queue_length": 0,
                "step_waiting_time": 0,
                "step_old_vehicle_ids": [],
            }

        num_vehicles = 0
        num_vehicles_out = 0

        while self.step < self.max_steps:
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                # If time to choose a new action
                if tl_state["green_time_remaining"] == 0:
                    state, base_green_time = self.get_state(tl)
                    action_idx = self.select_action(
                        tl_id,
                        state,
                        epsilon,
                    )

                    phase, green_delta = index_to_action(
                        action_idx,
                        self.actions_map[tl_id],
                    )

                    # Calculate green time like DQN
                    green_time = max(
                        base_green_time, self.green_time[tl_id] + green_delta
                    )
                    green_time = min(green_time, self.max_steps - self.step)

                    if tl_state["phase"] is not None:
                        self.set_yellow_phase(tl_id, tl_state["phase"])
                        for _ in range(self.interphase_duration):
                            self.accident_manager.create_accident(
                                current_step=self.step
                            )
                            traci.simulationStep()
                            num_vehicles += traci.simulation.getDepartedNumber()
                            self.step += 1

                            # Calculate step metrics during interphase for all traffic lights
                            for tl_inner in self.traffic_lights:
                                tl_inner_id = tl_inner["id"]
                                tl_inner_state = tl_states[tl_inner_id]

                                step_new_vehicle_ids = self.get_vehicles_in_phase(
                                    tl_inner,
                                    (
                                        tl_inner_state["phase"]
                                        if tl_inner_state["phase"] is not None
                                        else 0
                                    ),
                                )
                                step_outflow = sum(
                                    1
                                    for vid in tl_inner_state["step_old_vehicle_ids"]
                                    if vid not in step_new_vehicle_ids
                                )
                                tl_inner_state[
                                    "step_travel_speed_sum"
                                ] += self.get_sum_speed(tl_inner)
                                tl_inner_state[
                                    "step_travel_time_sum"
                                ] += self.get_sum_travel_time(tl_inner)
                                tl_inner_state[
                                    "step_density_sum"
                                ] += self.get_sum_density(tl_inner)
                                tl_inner_state["step_outflow"] += step_outflow
                                tl_inner_state[
                                    "step_queue_length"
                                ] += self.get_sum_queue_length(tl_inner)
                                tl_inner_state[
                                    "step_waiting_time"
                                ] += self.get_sum_waiting_time(tl_inner)
                                tl_inner_state["step_old_vehicle_ids"] = (
                                    step_new_vehicle_ids
                                )

                                # Check if it's time to save data (dynamic interval for exactly 60 data points)
                                if self.step % 60 == 0:
                                    travel_speed_avg = (
                                        tl_inner_state["step_travel_speed_sum"] / 60
                                    )
                                    travel_time_avg = (
                                        tl_inner_state["step_travel_time_sum"] / 60
                                    )
                                    density_avg = (
                                        tl_inner_state["step_density_sum"] / 60
                                    )
                                    outflow_avg = tl_inner_state["step_outflow"] / 60
                                    queue_length_avg = (
                                        tl_inner_state["step_queue_length"] / 60
                                    )
                                    waiting_time_avg = (
                                        tl_inner_state["step_waiting_time"] / 60
                                    )

                                    self.history["travel_speed"][tl_inner_id].append(
                                        travel_speed_avg
                                    )
                                    self.history["travel_time"][tl_inner_id].append(
                                        travel_time_avg
                                    )
                                    self.history["density"][tl_inner_id].append(
                                        density_avg
                                    )
                                    self.history["outflow"][tl_inner_id].append(
                                        outflow_avg
                                    )
                                    self.history["queue_length"][tl_inner_id].append(
                                        queue_length_avg
                                    )
                                    self.history["waiting_time"][tl_inner_id].append(
                                        waiting_time_avg
                                    )

                                    # Reset step metrics
                                    tl_inner_state["step_travel_speed_sum"] = 0
                                    tl_inner_state["step_travel_time_sum"] = 0
                                    tl_inner_state["step_density_sum"] = 0
                                    tl_inner_state["step_outflow"] = 0
                                    tl_inner_state["step_queue_length"] = 0
                                    tl_inner_state["step_waiting_time"] = 0

                    self.set_green_phase(tl_id, green_time, phase)

                    tl_state.update(
                        {
                            "green_time": green_time,
                            "green_time_remaining": green_time,
                            "travel_speed_sum": 0,
                            "travel_time_sum": 0,
                            "density_sum": 0,
                            "outflow": 0,
                            "old_vehicle_ids": self.get_vehicles_in_phase(tl, phase),
                            "state": state,
                            "action_idx": action_idx,
                            "phase": phase,
                        }
                    )

            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            self.step += 1

            for tl in self.traffic_lights:
                if self.step % 60 == 0:
                    tl_id = tl["id"]
                    tl_state = tl_states[tl_id]

                    travel_speed_avg = tl_state["step_travel_speed_sum"] / 60
                    travel_time_avg = tl_state["step_travel_time_sum"] / 60
                    density_avg = tl_state["step_density_sum"] / 60
                    outflow_avg = tl_state["step_outflow"] / 60
                    queue_length_avg = tl_state["step_queue_length"] / 60
                    waiting_time_avg = tl_state["step_waiting_time"] / 60

                    self.history["travel_speed"][tl_id].append(travel_speed_avg)
                    self.history["travel_time"][tl_id].append(travel_time_avg)
                    self.history["density"][tl_id].append(density_avg)
                    self.history["outflow"][tl_id].append(outflow_avg)
                    self.history["queue_length"][tl_id].append(queue_length_avg)
                    self.history["waiting_time"][tl_id].append(waiting_time_avg)

                    # Reset step metrics
                    tl_state["step_travel_speed_sum"] = 0
                    tl_state["step_travel_time_sum"] = 0
                    tl_state["step_density_sum"] = 0
                    tl_state["step_outflow"] = 0
                    tl_state["step_queue_length"] = 0
                    tl_state["step_waiting_time"] = 0

                if tl_state["green_time_remaining"] > 0:
                    phase = tl_state["phase"]
                    new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                    outflow = sum(
                        1
                        for vid in tl_state["old_vehicle_ids"]
                        if vid not in new_vehicle_ids
                    )
                    num_vehicles_out += outflow
                    tl_state["outflow"] += outflow
                    tl_state["travel_speed_sum"] += self.get_avg_speed(tl)
                    tl_state["travel_time_sum"] += self.get_avg_travel_time(tl)
                    tl_state["density_sum"] += self.get_avg_density(tl)
                    tl_state["old_vehicle_ids"] = new_vehicle_ids
                    tl_state["green_time_remaining"] -= 1

                    # When phase ends, store metrics and update Q-table
                    if tl_state["green_time_remaining"] == 0:
                        green_time = tl_state["green_time"]

                        # Calculate metrics like DQN
                        self.outflow_rate[tl_id] = tl_state["outflow"] / green_time
                        self.travel_speed[tl_id] = (
                            tl_state["travel_speed_sum"] / green_time
                        )
                        self.travel_time[tl_id] = (
                            tl_state["travel_time_sum"] / green_time
                        )
                        self.density[tl_id] = tl_state["density_sum"] / green_time

                        reward = self.get_reward(tl_id, tl_state["phase"])
                        next_state, _ = self.get_state(tl)
                        done = self.step >= self.max_steps

                        self.phase[tl_id] = tl_state["phase"]
                        
                        # Update green time tracking like DQN
                        self.green_time_old[tl_id] = self.green_time[tl_id]
                        self.green_time[tl_id] = green_time
                        
                        # Update old values for next reward calculation
                        self.old_outflow_rate[tl_id] = self.outflow_rate[tl_id]
                        self.old_travel_speed[tl_id] = self.travel_speed[tl_id]
                        self.old_travel_time[tl_id] = self.travel_time[tl_id]
                        self.old_density[tl_id] = self.density[tl_id]

                        self.agent_memory[tl_id].push(
                            tl_state["state"],
                            tl_state["action_idx"],
                            reward,
                            next_state,
                            done,
                        )

                        self.qlearn_update(
                            tl_state["state"],
                            tl_state["action_idx"],
                            reward,
                            next_state,
                            done,
                            tl_id,
                        )
                        self.history["agent_reward"][tl_id].append(reward)

                # step state metrics
                step_new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                step_outflow = sum(
                    1
                    for vid in tl_state["step_old_vehicle_ids"]
                    if vid not in step_new_vehicle_ids
                )
                tl_state["step_travel_speed_sum"] += self.get_avg_speed(tl)
                tl_state["step_travel_time_sum"] += self.get_avg_travel_time(tl)
                tl_state["step_density_sum"] += self.get_avg_density(tl)
                tl_state["step_outflow"] += step_outflow
                tl_state["step_queue_length"] += self.get_avg_queue_length(tl)
                tl_state["step_waiting_time"] += self.get_avg_waiting_time(tl)
                tl_state["step_old_vehicle_ids"] = step_new_vehicle_ids

        simulation_time = time.time() - start

        traci.close()

        print("Simulation ended")
        print("Number of vehicles:", num_vehicles)
        print("Number of vehicles get through all intersections:", num_vehicles_out)
        print("---------------------------------------")

        self.save_plot(episode=episode)
        self.step = 0

        total_reward = 0
        for traffic_light in self.traffic_lights:
            total_reward += np.sum(self.history["agent_reward"][traffic_light["id"]])

        print("Total reward:", total_reward, "- Epsilon:", epsilon)
        self.reset_history()
        # Decay epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        return simulation_time, 0

    def get_reward(self, traffic_light_id, phase=None):
        # Use DQN-style reward calculation (difference-based)
        return (
            self.weight["outflow_rate"]
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

    def save_q_table(self, path=None, episode=None):
        if path is None:
            path = self.path if hasattr(self, "path") else ""
        filename = (
            f"{path}q_table_episode_{episode}.pkl"
            if episode is not None
            else f"{path}q_table.pkl"
        )
        save_data = {
            'q_table': self.q_table,
            'rbf_weights': getattr(self, 'rbf_weights', {}),
            'rbf_sampler': getattr(self, 'rbf_sampler', None)
        }
        with open(filename, "wb") as f:
            pickle.dump(save_data, f)
        print(f"Q-table and RBF weights saved to {filename}")

    def load_q_table(self, path):
        """
        Load Q-table from the specified path.

        Args:
            path (str): Path to load the Q-table from.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                
            if isinstance(data, dict) and 'q_table' in data:
                # New format with RBF weights
                self.q_table = data['q_table']
                self.rbf_weights = data.get('rbf_weights', {})
                self.rbf_sampler = data.get('rbf_sampler', None)
                print(f"Q-table and RBF weights loaded from {path}")
            else:
                # Old format - just q_table
                self.q_table = data
                print(f"Q-table loaded from {path} (legacy format)")
        except FileNotFoundError:
            print(f"Q-table file not found: {path}. Starting with empty Q-table.")
        except Exception as e:
            print(f"Error loading Q-table: {e}. Starting with empty Q-table.")

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
            avg_history[metric] = avg_data

        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data(
                    data=data,
                    filename=f"q_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                )

            # Save Q-table and metrics DataFrame
            self.save_q_table(episode=episode)
            self.save_metrics_to_dataframe(episode=episode)

            print(
                "Plots, Q-table, and metrics DataFrame at episode", episode, "generated"
            )
            print("---------------------------------------")

    def save_metrics_to_dataframe(self, episode=None):
        """
        Save metrics per traffic light as pandas DataFrame.
        Only saves system metrics: density, outflow, queue_length, travel_speed, travel_time, waiting_time

        Returns:
            pd.DataFrame: DataFrame with columns [traffic_light_id, metric, time_step, value, episode]
        """
        data_records = []

        # Only collect specified system metrics
        target_metrics = [
            "density",
            "outflow",
            "queue_length",
            "travel_speed",
            "travel_time",
            "waiting_time",
        ]

        for metric, data_per_tls in self.history.items():
            if metric in target_metrics:
                for tl_id, data_list in data_per_tls.items():
                    if len(data_list) > 0:
                        for time_step, value in enumerate(data_list):
                            data_records.append(
                                {
                                    "traffic_light_id": tl_id,
                                    "metric": metric,
                                    "time_step": time_step,
                                    "value": value,
                                    "episode": episode,
                                    "simulation_type": "q_learning",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}q_learning_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}q_learning_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"Q-learning metrics DataFrame saved to {filename}")

        return df

    def get_yellow_phase(self, green_phase):
        """
        Convert a green phase string to a yellow phase string by replacing all 'G' with 'y'.
        """
        return green_phase.replace("G", "y")

    def set_yellow_phase(self, tlsId, green_phase):
        """
        Set the traffic light to yellow phase by converting green phase string to yellow.
        """
        yellow_phase = self.get_yellow_phase(green_phase)
        traci.trafficlight.setPhaseDuration(tlsId, 3)
        traci.trafficlight.setRedYellowGreenState(tlsId, yellow_phase)

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
        """
        Get the current state at a specific traffic light in the simulation.
        
        Returns:
            np.ndarray: 1D array representing the full input state
            int: green time (like DQN implementation)
        """
        state = [
            0,
            0,
            0,
            0,
            0,
            0,
        ]  # free_capacity, density, waiting_time, queue_length, phase_idx, green_time

        for detector in traffic_light["detectors"]:
            detector_id = detector["id"]
            free_capacity = self.get_free_capacity(detector_id)
            density = self.get_density(detector_id)
            waiting_time = self.get_waiting_time(detector_id)
            queue_length = self.get_queue_length(detector_id)

            state[0] += free_capacity
            state[1] += density
            state[2] = max(state[2], waiting_time)
            state[3] = max(state[3], queue_length)

        state[0] /= len(traffic_light["detectors"])
        state[1] /= len(traffic_light["detectors"])

        # Get green times and outflows from DESRA like DQN
        phase, green_time = self.desra.select_phase(traffic_light)

        state[4] = phase_to_index(phase, self.actions_map[traffic_light["id"]], 0)
        state[5] = green_time

        return np.array(state, dtype=np.float32), green_time

    def get_movements_from_phase(self, traffic_light, phase_str):
        detectors = [det["id"] for det in traffic_light["detectors"]]
        active_detectors = [
            detectors[i]
            for i, state in enumerate(phase_str)
            if state.upper() == "G" and i < len(detectors)
        ]
        return active_detectors

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_free_capacity(self, detector_id):
        """
        Get the free capacity of a lane.
        """
        lane_length = traci.lanearea.getLength(detector_id)
        occupied_length = (
            traci.lanearea.getLastStepOccupancy(detector_id) / 100 * lane_length
        )
        return (lane_length - occupied_length) / lane_length

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

    def get_sum_speed(self, traffic_light):
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                lane = traci.lanearea.getLaneID(detector["id"])
                max_speed = traci.lane.getMaxSpeed(lane)
                speeds.append(speed / max_speed)
            except:
                pass
        return np.sum(speeds) if speeds else 0.0

    def get_avg_speed(self, traffic_light):
        """
        Get average speed like DQN implementation.
        """
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                speeds.append(speed)
            except:
                pass
        return np.mean(speeds) if speeds else 0.0

    def get_sum_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

    def get_avg_travel_time(self, traffic_light):
        """
        Get average travel time like DQN implementation.
        """
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.mean(travel_times) if travel_times else 0.0

    def get_sum_density(self, traffic_light):
        densities = []
        for detector in traffic_light["detectors"]:
            try:
                densities.append(traci.lanearea.getLastStepOccupancy(detector["id"]))
            except:
                pass
        return np.sum(densities) / 100 if densities else 0.0

    def get_avg_density(self, traffic_light):
        """
        Get average density like DQN implementation.
        """
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

    def get_sum_queue_length(self, traffic_light):
        """
        Get the average queue length of a lane.
        """
        queue_lengths = []
        for detector in traffic_light["detectors"]:
            try:
                queue_lengths.append(
                    traci.lanearea.getLastStepVehicleNumber(detector["id"])
                )
            except:
                pass
        return np.sum(queue_lengths) if queue_lengths else 0.0

    def get_avg_queue_length(self, traffic_light):
        """
        Get the average queue length of a lane.
        """
        queue_lengths = []
        for detector in traffic_light["detectors"]:
            try:
                queue_lengths.append(traci.lanearea.getLastStepHaltingNumber(detector["id"]))
            except:
                pass
        return np.mean(queue_lengths) if queue_lengths else 0.0

    def get_sum_waiting_time(self, traffic_light):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = []
        for detector in traffic_light["detectors"]:
            vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(detector["id"]))

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vid)
        return total_waiting_time

    def get_avg_waiting_time(self, traffic_light):
        """
        Estimate average waiting time like DQN implementation.
        """
        vehicle_ids = []
        for detector in traffic_light["detectors"]:
            vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(detector["id"]))

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []

    def state_to_features(self, state, n_components=100, gamma=1.0):
        """
        Convert a continuous state vector to RBF features for function approximation.
        Uses sklearn's RBFSampler for feature mapping with consistent centers.

        Args:
            state: numpy array of continuous state values
            n_components: number of RBF features
            gamma: RBF kernel parameter

        Returns:
            np.ndarray: RBF feature vector
        """
        from sklearn.kernel_approximation import RBFSampler

        state = np.array(state).reshape(1, -1)
        
        # Initialize RBF sampler once with consistent centers
        if self.rbf_sampler is None:
            self.rbf_sampler = RBFSampler(
                n_components=self.n_components, 
                gamma=self.rbf_gamma, 
                random_state=42
            )
            # Fit on first state to establish centers
            self.rbf_sampler.fit(state)
        
        features = self.rbf_sampler.transform(state)
        return features.flatten()

    def get_q_rbf(self, state, action, n_components=100, gamma=1.0):
        """
        Get Q-value for state-action pair using RBF features and linear weights.
        """
        features = self.state_to_features(state)
        if action not in self.rbf_weights:
            self.rbf_weights[action] = np.zeros_like(features)
        return np.dot(self.rbf_weights[action], features)

    def set_q_rbf(
        self, state, action, target, alpha=None, n_components=100, gamma=1.0
    ):
        """
        Update Q-value weights for state-action pair using RBF features.
        """
        if alpha is None:
            alpha = self.alpha  # Use consistent learning rate
            
        features = self.state_to_features(state)
        if action not in self.rbf_weights:
            self.rbf_weights[action] = np.zeros_like(features)
        current = np.dot(self.rbf_weights[action], features)
        error = target - current
        self.rbf_weights[action] += alpha * error * features
        return self.rbf_weights[action]
