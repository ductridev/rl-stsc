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
    return actions_map[index]["phase"]


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

        self.step = 0
        self.num_actions = {}
        self.actions_map = {}

        self.agent_reward = {}
        self.agent_state = {}
        self.agent_old_state = {}
        self.agent_memory = {}

        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_speed = {}
        self.travel_time = {}
        self.waiting_time = {}
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
        self.q_tables = {tl["id"]: {} for tl in self.traffic_lights}
        self.alpha = 0.1  # learning rate
        self.gamma = self.agent_cfg["gamma"]
        self.epsilon = 1.0
        self.epsilon_min = self.agent_cfg.get("min_epsilon", 0.01)
        self.epsilon_decay = self.agent_cfg.get("decay_rate", 0.995)

        self.desra = DESRA(interphase_duration=self.interphase_duration)
        self.longest_phase, self.longest_phase_len, self.longest_phase_id = (
            self.get_longest_phase()
        )

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

            self.num_actions[traffic_light_id] = len(traffic_light["phase"])
            self.actions_map[traffic_light_id] = {}

            i = 0
            for phase in traffic_light["phase"]:
                self.actions_map[traffic_light_id][i] = {
                    "phase": phase,
                }
                i += 1

            self.agent_reward[traffic_light_id] = 0
            self.agent_state[traffic_light_id] = 0
            self.agent_old_state[traffic_light_id] = 0
            self.queue_length[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.travel_speed[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0
            self.phase[traffic_light_id] = None

            for key in self.history:
                self.history[key][traffic_light_id] = []

    def get_output_dims(self):
        output_dims = []
        for traffic_light in self.traffic_lights:
            output_dims.append(self.num_actions[traffic_light["id"]])
        return output_dims

    def discretize_state(self, state, precision=2):
        """Round each continuous feature so state tuples reuse similar values."""
        return tuple(np.round(state, precision).tolist())

    def get_q(self, state, action, tl_id):
        """Fetch Q[s,a], initializing the row if needed."""
        s_key = self.discretize_state(state)
        table = self.q_tables[tl_id]
        if s_key not in table:
            # initialize Q-values array for this new discretized state
            table[s_key] = np.zeros(self.num_actions[tl_id], dtype=np.float32)
        return table[s_key][action]

    def set_q(self, state, action, value, tl_id):
        """Set Q[s,a] after ensuring the row exists."""
        s_key = self.discretize_state(state)
        table = self.q_tables[tl_id]
        if s_key not in table:
            table[s_key] = np.zeros(self.num_actions[tl_id], dtype=np.float32)
        table[s_key][action] = value

    def select_action(self, tl_id, state, epsilon):
        """Epsilon-greedy using per-TL Q-table."""
        s_key = self.discretize_state(state)
        table = self.q_tables[tl_id]
        if s_key not in table:
            # unseen state: initialize and pick random
            table[s_key] = np.zeros(self.num_actions[tl_id], dtype=np.float32)
        if random.random() < epsilon:
            return random.randrange(self.num_actions[tl_id])
        return int(np.argmax(table[s_key]))

    def qlearn_update(self, state, action, reward, next_state, done, tl_id):
        """Standard Q-learning update, using discretized keys."""
        # get current Q
        old_q = self.get_q(state, action, tl_id)

        # compute max next Q
        if next_state is None or done:
            max_next = 0.0
        else:
            next_key = self.discretize_state(next_state)
            table = self.q_tables[tl_id]
            if next_key not in table:
                table[next_key] = np.zeros(self.num_actions[tl_id], dtype=np.float32)
            max_next = float(np.max(table[next_key]))

        # target and update
        target = reward + self.gamma * max_next
        new_q = old_q + self.alpha * (target - old_q)
        self.set_q(state, action, new_q, tl_id)

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start_time = time.time()

        # Initialize traffic light state tracking
        tl_states = self._init_traffic_light_states()

        num_vehicles = 0
        num_vehicles_out = 0

        while self.step < self.max_steps:
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                state = tl_states[tl_id]

                if state["green_time_remaining"] <= 0:
                    # 1. Select action
                    s = self.get_state(tl)
                    action_idx = self.select_action(tl_id, s, epsilon)
                    phase = index_to_action(action_idx, self.actions_map[tl_id])
                    green_time = min(state["green_time"], self.max_steps - self.step)

                    # 2. Run interphase
                    if state["phase"] is not None:
                        self._run_interphase_qlearn(tl_id, state["phase"], tl_states)

                    # 3. Set green phase
                    self.set_green_phase(tl_id, green_time, phase)
                    state.update(
                        {
                            "phase": phase,
                            "green_time": green_time,
                            "green_time_remaining": green_time,
                            "old_vehicle_ids": self.get_vehicles_in_phase(tl, phase),
                            "state": s,
                            "action_idx": action_idx,
                        }
                    )

            # === Global simulation step ===
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            self.step += 1

            # === Per-light update ===
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                state = tl_states[tl_id]

                if state["green_time_remaining"] > 0:
                    phase = state["phase"]
                    new_ids = self.get_vehicles_in_phase(tl, phase)
                    outflow = sum(
                        1 for vid in state["old_vehicle_ids"] if vid not in new_ids
                    )

                    num_vehicles_out += outflow
                    state["outflow"] += outflow
                    state["travel_speed_sum"] += self.get_sum_speed(tl)
                    state["travel_time_sum"] += self.get_sum_travel_time(tl)
                    state["waiting_time"] += self.get_sum_waiting_time(tl)
                    state["queue_length"] += self.get_sum_queue_length(tl)
                    state["old_vehicle_ids"] = new_ids
                    state["green_time_remaining"] -= 1

                    # Finalize phase when green ends
                    if state["green_time_remaining"] == 0:
                        self._finalize_phase_qlearn(tl, tl_id, state)

                # Always record step metrics
                self._record_step_metrics_qlearn(tl, tl_id, state)

                # Every 60 steps → flush metrics
                if self.step % 60 == 0:
                    self._flush_step_metrics_qlearn(tl, tl_id, state)

        # Post-simulation
        traci.close()
        sim_time = time.time() - start_time
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} through."
        )
        self.save_plot(episode=episode)
        self.step = 0

        # Total reward logging
        total_reward = sum(
            np.sum(self.history["agent_reward"][tl["id"]]) for tl in self.traffic_lights
        )
        print(f"Total reward: {total_reward}  -  Epsilon: {epsilon}")

        self.reset_history()
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return sim_time, 0

    def _init_traffic_light_states(self):
        states = {}
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            states[tl_id] = {
                "green_time_remaining": 20,
                "travel_speed_sum": 0,
                "travel_time_sum": 0,
                "waiting_time": 0,
                "outflow": 0,
                "queue_length": 0,
                "old_vehicle_ids": [],
                "state": None,
                "action_idx": None,
                "phase": tl["phase"][0],
                "green_time": 20,
                "step_travel_speed_sum": 0,
                "step_travel_time_sum": 0,
                "step_density_sum": 0,
                "step_outflow": 0,
                "step_queue_length": 0,
                "step_waiting_time": 0,
                "step_old_vehicle_ids": [],
            }
        return states

    def _run_interphase_qlearn(self, yellow_tl_id, old_phase, tl_states):
        for _ in range(self.interphase_duration):
            self.set_yellow_phase(yellow_tl_id, old_phase)
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            self.step += 1

            for tl in self.traffic_lights:
                tl_id = tl["id"]
                st = tl_states[tl_id]
                phase = st["phase"] or 0
                new_ids = self.get_vehicles_in_phase(tl, phase)
                outflow = sum(
                    1 for vid in st["step_old_vehicle_ids"] if vid not in new_ids
                )

                st["step_travel_speed_sum"] += self.get_sum_speed(tl)
                st["step_travel_time_sum"] += self.get_sum_travel_time(tl)
                st["step_density_sum"] += self.get_sum_density(tl)
                st["step_outflow"] += outflow
                st["step_queue_length"] += self.get_sum_queue_length(tl)
                st["step_waiting_time"] += self.get_sum_waiting_time(tl)
                st["step_old_vehicle_ids"] = new_ids

                if self.step % 60 == 0:
                    self._flush_step_metrics_qlearn(tl, tl_id, st)

    def _record_step_metrics_qlearn(self, tl, tl_id, st):
        new_ids = self.get_vehicles_in_phase(tl, st["phase"] or 0)
        outflow = sum(1 for vid in st["step_old_vehicle_ids"] if vid not in new_ids)
        st["step_old_vehicle_ids"] = new_ids

        st["step_travel_speed_sum"] += self.get_sum_speed(tl)
        st["step_travel_time_sum"] += self.get_sum_travel_time(tl)
        st["step_density_sum"] += self.get_sum_density(tl)
        st["step_outflow"] += outflow
        st["step_queue_length"] += self.get_sum_queue_length(tl)
        st["step_waiting_time"] += self.get_sum_waiting_time(tl)

    def _flush_step_metrics_qlearn(self, tl, tl_id, st):
        avg = lambda key: st[f"step_{key}"] / 60

        self.history["travel_speed"][tl_id].append(avg("travel_speed_sum"))
        self.history["travel_time"][tl_id].append(avg("travel_time_sum"))
        self.history["density"][tl_id].append(avg("density_sum"))
        self.history["outflow"][tl_id].append(avg("outflow"))
        self.history["queue_length"][tl_id].append(avg("queue_length"))
        self.history["waiting_time"][tl_id].append(avg("waiting_time"))

        for k in [
            "step_travel_speed_sum",
            "step_travel_time_sum",
            "step_density_sum",
            "step_outflow",
            "step_queue_length",
            "step_waiting_time",
        ]:
            st[k] = 0

    def _finalize_phase_qlearn(self, tl, tl_id, st):
        g = st["green_time"]
        self.queue_length[tl_id] = st["queue_length"]
        self.outflow_rate[tl_id] = st["outflow"] / g
        self.travel_speed[tl_id] = st["travel_speed_sum"]
        self.travel_time[tl_id] = st["travel_time_sum"] / g
        self.waiting_time[tl_id] = st["waiting_time"]

        reward = self.get_reward(tl_id, st["phase"])
        next_state = self.get_state(tl)
        done = self.step >= self.max_steps
        self.phase[tl_id] = st["phase"]

        if st["state"] is None:
            # first green phase: nothing to learn yet
            return

        self.agent_memory[tl_id].push(
            st["state"], st["action_idx"], reward, next_state, done
        )
        self.qlearn_update(
            st["state"], st["action_idx"], reward, next_state, done, tl_id
        )
        self.history["agent_reward"][tl_id].append(reward)

        # Reset counters
        for key in [
            "travel_speed_sum",
            "travel_time_sum",
            "outflow",
            "queue_length",
            "waiting_time",
        ]:
            st[key] = 0

    def get_reward(self, traffic_light_id, phase=None):
        return (
            self.weight["outflow_rate"]
            * self.outflow_rate_normalizer.normalize(
                self.outflow_rate[traffic_light_id]
            )
            + self.weight["delay"]
            * (
                1
                - self.travel_speed_normalizer.normalize(
                    self.travel_speed[traffic_light_id]
                )
            )
            + self.weight["waiting_time"]
            * self.waiting_time_normalizer.normalize(
                self.waiting_time[traffic_light_id]
            )
            + self.weight["switch_phase"] * (int)(self.phase[traffic_light_id] != phase)
            + self.weight["travel_time"]
            * self.travel_time_normalizer.normalize(self.travel_time[traffic_light_id])
            + self.weight["queue_length"]
            * self.queue_length_normalizer.normalize(
                self.queue_length[traffic_light_id]
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
        with open(filename, "wb") as f:
            pickle.dump(self.q_tables, f)
        print(f"Q-table saved to {filename}")

    def load_q_table(self, path):
        """
        Load Q-table from the specified path.

        Args:
            path (str): Path to load the Q-table from.
        """
        try:
            with open(path, "rb") as f:
                self.q_tables = pickle.load(f)
            print(f"Q-table loaded from {path}")
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
            int: green time
        """
        state_vector = []
        features_per_phase = 4  # free_capacity, density, waiting_time, queue_length
        # Compute max_phases from agent_cfg or fallback to number of phases
        max_phases = self.agent_cfg.get("max_phases", len(traffic_light["phase"]))

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)
            # Skip phases with no movements (safety check)
            if not movements:
                continue

            max_waiting_time = 0
            max_queue_length = 0
            for detector_id in movements:
                max_waiting_time = max(
                    max_waiting_time, self.get_waiting_time(detector_id)
                )
                max_queue_length = max(
                    max_queue_length, self.get_queue_length(detector_id)
                )

            avg_free_capacity = np.mean(
                [self.get_free_capacity(det) for det in movements]
            )
            avg_density = np.mean([self.get_density(det) for det in movements])

            # Append per-phase state: [free_capacity, density, waiting_time, queue_length]
            state_vector.extend(
                [avg_free_capacity, avg_density, max_waiting_time, max_queue_length]
            )
        # Calculate required input dim: max_phases * features_per_phase
        input_dim = max_phases * features_per_phase
        # Pad with zeros if needed
        padding = input_dim - len(state_vector)
        if padding > 0:
            state_vector.extend([0] * padding)

        return np.array(state_vector, dtype=np.float32)

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        length = traci.lanearea.getLength(detector_id)
        if length <= 0:
            return 0.0
        return traci.lanearea.getLastStepOccupancy(detector_id) / 100 * length

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

    def get_sum_speed(self, traffic_light):
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                lane = traci.lanearea.getLaneID(detector["id"])
                max_speed = traci.lane.getMaxSpeed(lane)
                speeds.append(speed / max_speed)
            except Exception as e:
                print(f"[WARN] Failed to get speed for {detector['id']}: {e}")
        return np.sum(speeds) if speeds else 0.0

    def get_sum_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

    def get_sum_density(self, traffic_light):
        """
        Compute density as total vehicles / total lane length (veh/m),
        using true vehicle counts from each detector.
        """
        total_veh = 0
        total_length = 0.0

        for det in traffic_light["detectors"]:
            try:
                count = traci.lanearea.getLastStepVehicleNumber(det["id"])
                lane_len = traci.lanearea.getLength(det["id"])
                total_veh += count
                total_length += lane_len
            except Exception:
                pass

        if total_length > 0:
            return total_veh / total_length
        else:
            return 0.0

    def get_num_phases(self, traffic_light):
        """
        Returns the number of custom-defined traffic light phases.
        """
        return len(traffic_light["phase"])

    def get_movements_from_phase(self, traffic_light, phase_str):
        """
        Get detector IDs whose street is active (green) in the given phase string.
        """
        phase_index = traffic_light["phase"].index(phase_str)  # Find index of phase_str
        active_street = str(
            phase_index + 1
        )  # Assuming street "1" is for phase 0, "2" is for phase 1, etc.

        # Collect detector IDs belonging to the active street
        active_detectors = [
            det["id"]
            for det in traffic_light["detectors"]
            if det["street"] == active_street
        ]

        return active_detectors

    def get_sum_queue_length(self, traffic_light):
        """
        Get the average queue length of a lane.
        """
        queue_lengths = []
        for detector in traffic_light["detectors"]:
            length = traci.lanearea.getLength(detector["id"])
            if length <= 0:
                queue_lengths.append(0.0)
            else:
                queue_lengths.append(
                    traci.lanearea.getLastStepOccupancy(detector["id"]) / 100 * length
                )
        return np.sum(queue_lengths) if queue_lengths else 0.0

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
        return np.tanh(total_waiting_time / 100)

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
