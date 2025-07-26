from src.model import DQN
from src.memory import ReplayMemory
from src.visualization import Visualization
from src.normalizer import Normalizer
from src.desra import DESRA
from src.sumo import SUMO
from src.accident_manager import AccidentManager
import torch.nn.functional as F
import traci
import numpy as np
import random
import torch
import time
import torch.nn as nn
import copy
import pandas as pd
from collections import defaultdict, deque

GREEN_ACTION = 0
RED_ACTION = 1


def index_to_action(index, actions_map):
    return actions_map[index]["phase"]


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
        self.loss_type = agent_cfg["loss_type"]
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
        # buffer raw counts (veh per step) with timestamps
        self.arrival_buffers = defaultdict(lambda: deque())
        # remember last time we ran DESRA for each detector
        self.last_desra_time = {}

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

        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Training will run on CPU.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.agent: DQN = DQN(
            num_layers=self.agent_cfg["num_layers"],
            batch_size=self.agent_cfg["batch_size"],
            learning_rate=self.agent_cfg["learning_rate"],
            input_dim=self.agent_cfg["num_states"],
            output_dims=self.get_output_dims(),
            gamma=self.agent_cfg["gamma"],
            device=self.device,
            loss_type=self.loss_type,
        )

        self.desra = DESRA(interphase_duration=self.interphase_duration)

        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs")
            self.agent = nn.DataParallel(self.agent)

        self.agent = self.agent.to(self.device)
        self.target_net = copy.deepcopy(self.agent)
        self.target_net.eval()

        self.longest_phase, self.longest_phase_len, self.longest_phase_id = (
            self.get_longest_phase()
        )

        self.tl_states = {}

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

            # Initialize the number of actions
            self.num_actions[traffic_light_id] = len(traffic_light["phase"])

            # Initialize the action map
            self.actions_map[traffic_light_id] = {}

            # We will convert from number of phases and green delta to phase index and green delta
            i = 0

            for phase in traffic_light["phase"]:
                self.actions_map[traffic_light_id][i] = {
                    "phase": phase,
                }

                i += 1

            # Initialize the agent reward
            self.agent_reward[traffic_light_id] = 0

            # Initialize the agent state
            self.agent_state[traffic_light_id] = 0

            # Initialize the agent old state
            self.agent_old_state[traffic_light_id] = 0

            # Initialize the queue length
            self.queue_length[traffic_light_id] = 0

            # Initialize the outflow rate
            self.outflow_rate[traffic_light_id] = 0

            # Initialize the travel speed
            self.travel_speed[traffic_light_id] = 0

            # Initialize the travel time
            self.travel_time[traffic_light_id] = 0

            # Initialize the density
            self.waiting_time[traffic_light_id] = 0

            # Initialize the phase
            self.phase[traffic_light_id] = None

            # Initialize the history
            for key in self.history:
                self.history[key][traffic_light_id] = []

            # Initialize the arrival buffers
            for tl in self.traffic_lights:
                for phase in tl["phase"]:
                    for det in self.get_movements_from_phase(tl, phase):
                        # force the key to exist
                        _ = self.arrival_buffers[det]

    def get_output_dims(self):
        """
        Get multiple outputs of the agent for each traffic light

        Returns:
            output_dims (list[int]): The output dimension of the agent
        """

        output_dims = []
        for traffic_light in self.traffic_lights:
            output_dims.append(self.num_actions[traffic_light["id"]])

        return output_dims

    def run(self, epsilon, episode):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        # build a flat list of all detector IDs once
        self.all_detectors = [
            det
            for tl in self.traffic_lights
            for phase_str in tl["phase"]
            for det in self.get_movements_from_phase(tl, phase_str)
        ]
        # initialize last_desra_time
        for det in self.all_detectors:
            self.last_desra_time[det] = traci.simulation.getTime()

        start_time = time.time()

        # Initialize per-light state tracking
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            self.tl_states[tl_id] = {
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
                "step_outflow_sum": 0,
                "step_density_sum": 0,
                "step_queue_length_sum": 0,
                "step_waiting_time_sum": 0,
                "step_old_vehicle_ids": [],
            }

        num_vehicles = 0
        num_vehicles_out = 0

        while self.step < self.max_steps:
            # === 1) Action selection (when green expires) ===
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                state = self.tl_states[tl_id]
                if state["green_time_remaining"] <= 0:
                    # 1a) pick new action
                    s = self.get_state(tl)
                    random_val, action_idx, pred_green = self.select_action(
                        tl_id, self.agent, s, epsilon
                    )
                    phase = index_to_action(action_idx, self.actions_map[tl_id])
                    desra_action_idx, desra_green_time = s[-2:]
                    if (
                        desra_action_idx != action_idx
                        and desra_green_time != pred_green
                    ):
                        desra_phase = index_to_action(
                            desra_action_idx, self.actions_map[tl_id]
                        )
                        print(
                            f"[DEBUG] Random value: {random_val} - DQN Phase: {phase} - DQN Green time: {pred_green} - DESRA Phase: {desra_phase} - DESRA Green time: {s[-1:][0]}"
                        )
                    green_time = max(1, min(pred_green, self.max_steps - self.step))

                    # 1b) if we had a previous phase, run its yellow interphase
                    if state["phase"] is not None:
                        self._run_interphase(tl_id, state["phase"])

                    # 1c) switch this TL to its new green
                    self.set_green_phase(tl_id, green_time, phase)
                    state.update(
                        {
                            "phase": phase,
                            "green_time": green_time,
                            "green_time_remaining": green_time,
                            "old_vehicle_ids": self.get_vehicles_in_phase(tl, phase),
                        }
                    )

            # === 2) Global simulation step ===
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            self.step += 1
            self._record_arrivals()

            # === 3) Metric collection for each TL ===
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                st = self.tl_states[tl_id]
                phase = st["phase"]

                # a) per‐step outflow
                new_ids = self.get_vehicles_in_phase(tl, phase)
                outflow = sum(1 for vid in st["old_vehicle_ids"] if vid not in new_ids)
                st["old_vehicle_ids"] = new_ids

                # b) accumulate
                num_vehicles_out += outflow
                st["step_outflow_sum"] += outflow
                st["step_travel_speed_sum"] += self.get_sum_speed(tl)
                st["step_travel_time_sum"] += self.get_sum_travel_time(tl)
                st["step_density_sum"] += self.get_sum_density(tl)
                st["step_queue_length_sum"] += self.get_sum_queue_length(tl)
                st["step_waiting_time_sum"] += self.get_sum_waiting_time(tl)

                # c) countdown green
                if st["green_time_remaining"] > 0:
                    st["green_time_remaining"] -= 1
                else:
                    # when it just expired, record overall metrics & push to memory
                    self._finalize_phase(tl, tl_id, st)

                # d) every 60 steps flush partial metrics into history
                if self.step % 60 == 0:
                    self._flush_step_metrics(tl, tl_id, st)

        traci.close()
        sim_time = time.time() - start_time
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} through."
        )
        return sim_time, self._train_and_plot(epsilon, episode)

    def _record_arrivals(self):
        """
        Call this exactly once per traci.simulationStep().
        It loops all detectors and pushes the raw count upstream,
        so we can later return a moving average in get_arrival_flow.
        """
        t = traci.simulation.getTime()
        for tl in self.traffic_lights:
            for phase_str in tl["phase"]:
                for det in self.get_movements_from_phase(tl, phase_str):
                    lane_id = traci.lanearea.getLaneID(det)
                    incoming = [
                        link[0]
                        for link in traci.lane.getLinks(lane_id)
                        if link[0] != lane_id
                    ]
                    raw_count = sum(
                        traci.lane.getLastStepVehicleNumber(l) for l in incoming
                    )
                    self.arrival_buffers[det].append((t, raw_count))

    def _compute_arrival_flows(self):
        """
        Turn each detector’s deque of (t,count) into a veh/s over
        the exact interval since the last DESRA call.
        """
        now = traci.simulation.getTime()
        q_arrivals = {}
        for det, buf in self.arrival_buffers.items():
            t0 = self.last_desra_time[det]
            # only keep entries newer than t0
            recent = [(ts, c) for ts, c in buf if ts > t0]
            total_veh = sum(c for ts, c in recent)
            dt = max(now - t0, 1e-6)
            q_arrivals[det] = total_veh / dt
            # purge old entries to free memory
            while buf and buf[0][0] <= t0:
                buf.popleft()
            # record next baseline
            self.last_desra_time[det] = now
        return q_arrivals

    def _train_and_plot(self, epsilon, episode):
        """
        Swaps in your existing training & plotting code.
        Returns training_time.
        """
        print("Training...")
        start_time = time.time()

        # update target net every 10 episodes
        if episode % 10 == 0:
            self.target_net.load_state_dict(self.agent.state_dict())

        # run your training epochs
        for _ in range(self.epoch):
            self.training()

        training_time = time.time() - start_time
        print("Training ended")
        print("---------------------------------------")

        # save episode plot
        self.save_plot(episode=episode)

        # reset step counter & history
        self.step = 0

        # compute and log total reward
        total_reward = 0
        for tl in self.traffic_lights:
            rewards = self.history["agent_reward"].get(tl["id"], [])
            total_reward += np.sum(rewards)
        print(f"Total reward: {total_reward}  -  Epsilon: {epsilon}")

        self.reset_history()
        return training_time

    def _run_interphase(self, yellow_tl_id, old_phase):
        """
        Run exactly self.interphase_duration steps where only
        `yellow_tl_id` is set to yellow; all others keep their last phase.
        Collect metrics in those steps.
        """
        for _ in range(self.interphase_duration):
            self.set_yellow_phase(yellow_tl_id, old_phase)
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            self.step += 1

            # collect per-step metrics on *all* traffic lights
            for tl in self.traffic_lights:
                st = self.tl_states[tl["id"]]
                phase = st["phase"] or 0
                new_ids = self.get_vehicles_in_phase(tl, phase)
                outflow = sum(
                    1 for vid in st["step_old_vehicle_ids"] if vid not in new_ids
                )
                st["step_old_vehicle_ids"] = new_ids

                st["step_outflow_sum"] += outflow
                st["step_travel_speed_sum"] += self.get_sum_speed(tl)
                st["step_travel_time_sum"] += self.get_sum_travel_time(tl)
                st["step_density_sum"] += self.get_sum_density(tl)
                st["step_queue_length_sum"] += self.get_sum_queue_length(tl)
                st["step_waiting_time_sum"] += self.get_sum_waiting_time(tl)

                if self.step % 60 == 0:
                    self._flush_step_metrics(tl, tl["id"], st)

    def _flush_step_metrics(self, tl, tl_id, st):
        """
        Every 60 steps, average the step accumulators and append
        to self.history, then zero them out.
        """
        # Helper to compute 60-step average
        avg = lambda name: st[f"step_{name}_sum"] / 60

        # 1) Normalize outflow to per-lane basis
        total_outflow = avg("outflow")
        # Count movements (detectors) under this TL
        movements = [
            det
            for phase in tl["phase"]
            for det in self.get_movements_from_phase(tl, phase)
        ]
        num_movements = max(len(movements), 1)
        norm_outflow = total_outflow / num_movements
        self.history["outflow"][tl_id].append(norm_outflow)
        st["step_outflow_sum"] = 0

        # 2) Record true density (veh/m) directly
        corrected_density = self.get_sum_density(tl)
        self.history["density"][tl_id].append(corrected_density)
        st["step_density_sum"] = 0

        # 3) Append other metrics as before
        for metric in ["travel_speed", "travel_time", "queue_length", "waiting_time"]:
            val = avg(metric)
            self.history[metric][tl_id].append(val)
            st[f"step_{metric}_sum"] = 0

    def _finalize_phase(self, tl, tl_id, st):
        """
        Called once, when a green expires.  Records the overall
        metrics for that full green, pushes to replay memory, etc.
        """
        g = st["green_time"]
        self.queue_length[tl_id] = st["queue_length"]
        self.outflow_rate[tl_id] = st["outflow"] / g
        self.travel_speed[tl_id] = st["travel_speed_sum"]
        self.travel_time[tl_id] = st["travel_time_sum"] / g
        self.waiting_time[tl_id] = st["waiting_time"]

        reward = self.get_reward(tl_id, st["phase"])
        next_state = self.get_state(tl)
        done = self.step >= self.max_steps

        self.agent_memory[tl_id].push(
            st["state"], st["action_idx"], g, reward, next_state, done
        )
        self.history["agent_reward"][tl_id].append(reward)

        # reset for next green
        for key in [
            "travel_speed_sum",
            "travel_time_sum",
            "density_sum",
            "outflow",
            "queue_length",
            "waiting_time",
        ]:
            st[key] = 0

    def estimate_fd_params(self, tl_id, window=60, min_history=10, ema_alpha=None):
        """
        Estimate per-lane saturation_flow, critical_density, jam_density for a traffic light
        using recent flow and density history.
        
        Args:
            tl_id (str): Traffic light ID
            window (int): How many entries of history to use
            min_history (int): Minimum number of points before trusting empirical data
            ema_alpha (float or None): If set, apply EMA smoothing with this alpha 
                                    against previous self.saturation_flow, etc.
        
        Returns:
            (saturation_flow, critical_density, jam_density)
        """
        # Grab the last `window` points
        flow_hist = self.history["outflow"].get(tl_id, [])[-window:]
        density_hist = self.history["density"].get(tl_id, [])[-window:]
        
        # Not enough data: use defaults
        if len(flow_hist) < min_history or len(density_hist) < min_history:
            return 0.5, 0.06, 0.18
        
        flow_arr = np.array(flow_hist)
        dens_arr = np.array(density_hist)
        
        # 1) Saturation flow: use the 90th percentile to avoid spikes
        s_emp = np.percentile(flow_arr, 90)
        
        # 2) Critical density: density at the 90th percentile of flow
        threshold = np.percentile(flow_arr, 90)
        # pick the density corresponding to the first time flow crosses that threshold
        idx = np.argmax(flow_arr >= threshold)
        k_c_emp = dens_arr[idx]
        
        # 3) Jam density: use a high percentile to avoid a single noise spike
        k_j_emp = np.percentile(dens_arr, 95)
        
        # 4) Ensure jam > critical by at least 20%
        if k_j_emp <= k_c_emp * 1.2:
            k_j_emp = k_c_emp * 1.2 + 1e-3
        
        # 5) Optional EMA smoothing
        if ema_alpha is not None:
            # initialize on first call
            current_s = getattr(self, "_ema_saturation_flow", s_emp)
            current_kc = getattr(self, "_ema_critical_density", k_c_emp)
            current_kj = getattr(self, "_ema_jam_density", k_j_emp)
            
            s_new  = (1-ema_alpha)*current_s  + ema_alpha*s_emp
            kc_new = (1-ema_alpha)*current_kc + ema_alpha*k_c_emp
            kj_new = (1-ema_alpha)*current_kj + ema_alpha*k_j_emp
            
            # store for next call
            self._ema_saturation_flow  = s_new
            self._ema_critical_density = kc_new
            self._ema_jam_density      = kj_new
            
            return s_new, kc_new, kj_new
        
        return s_emp, k_c_emp, k_j_emp

    def training(self):
        """
        Retrieve a batch from each traffic light memory and train the agent.
        """
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            batch = self.agent_memory[traffic_light_id].get_samples(
                self.agent.batch_size
            )

            if len(batch) > 0:
                state_data, rewards, next_state_data, dones = zip(*batch)
                states, actions, green_times = zip(*state_data)
                next_states, _ = zip(*next_state_data)

                metrics = self.agent.train_batch(
                    states,
                    actions,
                    rewards,
                    next_states,
                    green_targets=green_times,
                    output_dim=self.num_actions[traffic_light_id],
                    done=dones,
                    target_net=self.target_net,
                )

                self.history["q_value"][traffic_light_id].append(metrics["avg_q_value"])
                self.history["max_next_q_value"][traffic_light_id].append(
                    metrics["avg_max_next_q_value"]
                )
                self.history["target"][traffic_light_id].append(metrics["avg_target"])
                self.history["loss"][traffic_light_id].append(metrics["total_loss"])

    def get_reward(self, traffic_light_id, phase):
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

    def save_plot(self, episode):

        # Average history over all traffic lights
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

        # Save and plot averaged metrics
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data(
                    data=data,
                    filename=f"dqn_{self.loss_type}_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                )

            # Save metrics as DataFrame
            self.save_metrics_to_dataframe(episode=episode)

            print("Plots at episode", episode, "generated")
            print("---------------------------------------")
            # Reset history after saving plots

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
                                    "simulation_type": f"dqn_{self.loss_type}",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}dqn_{self.loss_type}_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}dqn_{self.loss_type}_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"DQN {self.loss_type} metrics DataFrame saved to {filename}")

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

    def select_action(
        self,
        traffic_light_id: str,
        agent: DQN,
        base_state: np.ndarray,
        epsilon: float,
    ):
        """
        Select an action using epsilon-greedy policy with DESRA green time hints.

        Args:
            traffic_light_id (str): ID of the traffic light
            agent (DQN): The DQN model
            base_state (np.ndarray): Base traffic state (e.g. queue, arrivals)
            epsilon (float): Epsilon-greedy exploration factor

        Returns:
            (int, float): Tuple of selected phase index and predicted green time
        """
        num_actions = self.num_actions[traffic_light_id]

        # Add DESRA green hints to the state
        state_t = torch.from_numpy(base_state).to(self.device, dtype=torch.float32)

        random_val = random.uniform(0, 1)

        if epsilon == self.agent_cfg["epsilon"]:
            # Use DESRA phase and green time hints
            desra_phase = int(base_state[-2])
            desra_green = float(base_state[-1])

            return random_val, desra_phase, desra_green

        if random_val < epsilon:
            # Explore using DESRA phase but add green time randomness
            desra_phase = int(base_state[-2])
            desra_green = float(base_state[-1])

            # Add variation to DESRA green time (e.g. ±5 seconds)
            variation = random.uniform(-5, 5)
            green_time = max(5, desra_green + variation)  # clamp min green time

            return random_val, desra_phase, green_time

        with torch.no_grad():
            q_values, green_times = agent.predict_one(state_t, output_dim=num_actions)
            if agent.loss_type == "qr":
                q_values = q_values.mean(2)  # [1, A]

            best_action_idx = q_values.squeeze(0).argmax().item()
            predicted_green_time = green_times.squeeze(0)[best_action_idx].item()

            return random_val, best_action_idx, predicted_green_time

    def get_state(self, traffic_light):
        """
        Get the current state at a specific traffic light in the simulation.

        Returns:
            np.ndarray: 1D array representing the full input state
            int: green time
        """
        state_vector = []

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

        #  Compute per‐detector q_arr over [last_call, now]
        q_arr_dict = self._compute_arrival_flows()

        saturation_flow, critical_density, jam_density = self.estimate_fd_params(
            traffic_light["id"]
        )

        # DESRA recommended phase and green time
        best_phase, desra_green = self.desra.select_phase_with_desra_hints(
            traffic_light,
            q_arr_dict,
            saturation_flow=saturation_flow,
            critical_density=critical_density,
            jam_density=jam_density,
        )

        padding = self.agent._input_dim - len(state_vector) - 2
        state_vector.extend([0] * padding)

        # Convert phase to index
        desra_phase_idx = phase_to_index(
            best_phase, self.actions_map[traffic_light["id"]], 0
        )

        # Append DESRA guidance
        state_vector.extend([desra_phase_idx, desra_green])

        assert (
            len(state_vector) == self.agent._input_dim
        ), f"State vector length {len(state_vector)} does not match input_dim {self.agent._input_dim}"

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
