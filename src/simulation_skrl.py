from skrl.agents.torch.dqn import DQN as SKRL_DQN, DQN_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

from src.visualization import Visualization
from src.normalizer import Normalizer
from src.desra import DESRA
from src.sumo import SUMO
from src.accident_manager import AccidentManager
from src.vehicle_tracker import VehicleTracker
from src.traffic_light_env import TrafficLightEnv, TrafficLightEnvFactory
from src.model_skrl import ModelFactory
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


class SimulationSKRL(SUMO):
    def __init__(
        self,
        visualization: Visualization,
        agent_cfg,
        max_steps,
        traffic_lights,
        accident_manager: AccidentManager,
        interphase_duration=3,
        epoch=1000,
        path=None,
        training_steps=300,
        updating_target_network_steps=100,
        save_interval=2,
    ):
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
        self.training_steps = training_steps
        self.updating_target_network_steps = updating_target_network_steps
        self.save_interval = save_interval

        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_delay_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()

        # Initialize VehicleTracker for logging vehicle statistics
        self.vehicle_tracker = VehicleTracker(path=self.path)

        self.step = 0
        self.num_actions = {}
        self.actions_map = {}

        self.agent_reward = {}
        self.agent_state = {}
        self.agent_old_state = {}
        
        # SKRL-specific memory and agents
        self.memories = {}
        self.agents = {}
        self.envs = {}
        
        # buffer raw counts (veh per step) with timestamps
        self.arrival_buffers = defaultdict(lambda: deque())
        # remember last time we ran DESRA for each detector
        self.last_desra_time = {}

        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_delay = {}
        self.travel_time = {}
        self.waiting_time = {}
        self.phase = {}

        self.history = {
            "agent_reward": {},
            "travel_delay": {},
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

        # Initialize SKRL components
        self._init_skrl_components()

        self.desra = DESRA(interphase_duration=self.interphase_duration)

        self.longest_phase, self.longest_phase_len, self.longest_phase_id = (
            self.get_longest_phase()
        )

        self.tl_states = {}

    def _init_skrl_components(self):
        """Initialize SKRL DQN agents and memories for each traffic light"""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        for traffic_light in self.traffic_lights:
            tl_id = traffic_light["id"]
            
            # Create environment for this traffic light using factory
            env = TrafficLightEnvFactory.create_single_env(self, tl_id)
            self.envs[tl_id] = env
            
            # Create memory
            memory = RandomMemory(
                memory_size=self.agent_cfg.get("memory_size_max", 10000),
                num_envs=1,
                device=self.device
            )
            self.memories[tl_id] = memory
            
            # Create models using factory
            model_kwargs = {
                "num_layers": self.agent_cfg["num_layers"],
                "hidden_size": self.agent_cfg.get("hidden_size", 256),
                "dropout_rate": self.agent_cfg.get("dropout_rate", 0.1),
                "use_batch_norm": self.agent_cfg.get("use_batch_norm", False),
                "use_dueling": self.agent_cfg.get("use_dueling", False)
            }
            
            models = ModelFactory.create_dqn_models(
                env.observation_space, 
                env.action_space, 
                self.device,
                **model_kwargs
            )
            
            # DQN configuration using SKRL default config
            cfg_dqn = DQN_DEFAULT_CONFIG.copy()
            cfg_dqn["batch_size"] = int(self.agent_cfg["batch_size"])
            cfg_dqn["discount_factor"] = float(self.agent_cfg["gamma"])
            cfg_dqn["learning_rate"] = float(self.agent_cfg["learning_rate"])
            cfg_dqn["target_update_frequency"] = int(self.updating_target_network_steps)
            cfg_dqn["exploration"]["initial_epsilon"] = float(self.agent_cfg.get("epsilon", 1.0))
            cfg_dqn["exploration"]["final_epsilon"] = float(self.agent_cfg.get("min_epsilon", 0.01))
            cfg_dqn["exploration"]["timesteps"] = int(self.agent_cfg.get("exploration_timesteps", 50000))
            cfg_dqn["learning_starts"] = int(self.agent_cfg.get("learning_starts", 1000))
            cfg_dqn["double_dqn"] = True
            
            # Helper function to safely convert config values
            def safe_int_convert(value, default=0):
                if isinstance(value, str) and value.lower() == 'auto':
                    return default
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return default
            
            # Ensure critical config values are correct types
            cfg_dqn["checkpoint_interval"] = safe_int_convert(cfg_dqn.get("checkpoint_interval", 0))
            
            # Handle experiment config safely
            if "experiment" in cfg_dqn:
                cfg_dqn["experiment"]["write_interval"] = safe_int_convert(
                    cfg_dqn["experiment"].get("write_interval", 0)
                )
                cfg_dqn["experiment"]["checkpoint_interval"] = safe_int_convert(
                    cfg_dqn["experiment"].get("checkpoint_interval", 0)
                )
            
            # Convert any other potential string values in the config
            for key in ["grad_norm_clip", "polyak"]:
                if key in cfg_dqn:
                    if isinstance(cfg_dqn[key], str):
                        cfg_dqn[key] = 0.0 if cfg_dqn[key].lower() == 'auto' else float(cfg_dqn[key])
            
            # Create DQN agent
            agent = SKRL_DQN(
                models=models,
                memory=memory,
                cfg=cfg_dqn,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=self.device
            )
            
            # Initialize the agent (this sets up tensors_names and other attributes)
            agent.init(trainer_cfg=None)
            
            self.agents[tl_id] = agent

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

            # Initialize the travel delay
            self.travel_delay[traffic_light_id] = 0

            # Initialize the travel time
            self.travel_time[traffic_light_id] = 0

            # Initialize the waiting time
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
        print("Simulation started (SKRL)")
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
                "green_time_remaining": 5,
                "yellow_time_remaining": 0,
                "interphase": False,
                "travel_delay_sum": 0,
                "travel_time_sum": 0,
                "waiting_time": 0,
                "outflow": 0,
                "queue_length": 0,
                "old_vehicle_ids": [],
                "state": None,
                "action_idx": None,
                "phase": tl["phase"][0],
                "old_phase": tl["phase"][0],
                "green_time": 5,
                "step_travel_delay_sum": 0,
                "step_travel_time_sum": 0,
                "step_outflow_sum": 0,
                "step_density_sum": 0,
                "step_queue_length_sum": 0,
                "step_waiting_time_sum": 0,
                "step_old_vehicle_ids": [],
                "observation": None,
            }

        num_vehicles = 0
        num_vehicles_out = 0

        # Reset all environments
        for tl_id in self.envs:
            self.envs[tl_id].reset()

        while self.step < self.max_steps:
            # === 1) Action selection (when green expires) ===
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                state = self.tl_states[tl_id]

                # 1a) if we're still in yellow countdown, just tick it down
                if state["yellow_time_remaining"] > 0:
                    state["yellow_time_remaining"] -= 1
                    continue

                # 1b) if green still running, skip
                if state["green_time_remaining"] > 0:
                    continue

                # 1c) green just ended, if old phase is different with new phase enter yellow and if not already in interphase
                if not state["interphase"] and state["old_phase"] != state["phase"]:
                    self.set_yellow_phase(tl_id, state["phase"])
                    state["yellow_time_remaining"] = self.interphase_duration
                    state["interphase"] = True
                    continue

                # 1d) green just finished → pick new action
                # reset interphase flag now
                state["interphase"] = False

                # Get state
                s = self.get_state(tl, state["phase"])
                state["observation"] = s

                # Select action using SKRL agent
                action_idx = self._select_action_skrl(tl_id, s, epsilon)

                # Convert action to phase
                new_phase = index_to_action(action_idx, self.actions_map[tl_id])

                # Get green time to not exceed max_steps
                green_time = max(
                    1, min(state["green_time"], self.max_steps - self.step)
                )

                # switch to new green
                self.set_green_phase(tl_id, green_time, new_phase)

                # Update state
                state.update(
                    {
                        "phase": new_phase,
                        "old_phase": state["phase"],
                        "green_time": green_time,
                        "green_time_remaining": green_time,
                        "old_vehicle_ids": self.get_vehicles_in_phase(tl, new_phase),
                        "state": s,
                        "action_idx": action_idx,
                    }
                )

            # === 2) Global simulation step ===
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            num_vehicles_out += traci.simulation.getArrivedNumber()
            self.step += 1

            # Update vehicle tracking statistics
            self.vehicle_tracker.update_stats(self.step)

            # Print vehicle stats every 1000 steps
            if self.step % 1000 == 0:
                current_stats = self.vehicle_tracker.get_current_stats()
                print(
                    f"Step {self.step}: "
                    f"Running={current_stats['total_running']}, "
                    f"Total Departed={current_stats['total_departed']}, "
                    f"Total Arrived={current_stats['total_arrived']}"
                )

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

                sum_travel_delay = self.get_sum_travel_delay(tl)
                sum_travel_time = self.get_sum_travel_time(tl)
                sum_density = self.get_sum_density(tl)
                sum_queue_length = self.get_sum_queue_length(tl)
                sum_waiting_time = self.get_sum_waiting_time(tl)

                # b) accumulate
                # Update metrics for per 60 steps
                st["step_outflow_sum"] += outflow
                st["step_travel_delay_sum"] += sum_travel_delay
                st["step_travel_time_sum"] = sum_travel_time
                st["step_density_sum"] += sum_density
                st["step_queue_length_sum"] += sum_queue_length
                st["step_waiting_time_sum"] = sum_waiting_time

                # Update metrics for current phase
                st["outflow"] += outflow
                st["travel_delay_sum"] = sum_travel_delay
                st["travel_time_sum"] = sum_travel_time
                st["queue_length"] = sum_queue_length
                st["waiting_time"] = sum_waiting_time

                # c) countdown green
                if st["green_time_remaining"] > 0:
                    st["green_time_remaining"] -= 1
                else:
                    # when it just expired, record overall metrics & push to memory
                    self._finalize_phase_skrl(tl, tl_id, st)

                # d) every 60 steps flush partial metrics into history
                if self.step > 0 and self.step % 60 == 0:
                    self._flush_step_metrics(tl, tl_id, st)

                # SKRL training
                if self.step > 0 and self.step % self.training_steps == 0:
                    print(f"SKRL Training per {self.training_steps} steps...")
                    start = time.time()
                    self._train_skrl_agents()
                    print(
                        f"SKRL Training per {self.training_steps} steps took {time.time() - start}"
                    )

        traci.close()
        sim_time = time.time() - start_time
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} through."
        )

        # Total reward logging
        total_reward = sum(
            np.sum(self.history["agent_reward"][tl["id"]]) for tl in self.traffic_lights
        )
        print(f"Total reward: {total_reward}  -  Epsilon: {epsilon}")

        # Clear agent memory (SKRL RandomMemory doesn't have clear method)
        # Memory will be automatically managed by SKRL agents

        # Print and save vehicle statistics
        self.vehicle_tracker.print_summary("dqn_skrl")
        self.vehicle_tracker.save_logs(episode, "dqn_skrl")
        # Note: vehicle_tracker.reset() moved to train.py after performance tracking

        return sim_time, self._train_and_plot(epsilon, episode)

    def _select_action_skrl(self, tl_id, state, epsilon):
        """
        Select action using SKRL DQN agent
        """
        agent = self.agents[tl_id]
        
        # Convert state to tensor (ensure it's the right shape)
        if isinstance(state, (list, np.ndarray)):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device) if state.dim() == 1 else state.to(self.device)
        
        # Use agent's act method - SKRL expects states tensor directly
        with torch.no_grad():
            # Pass states tensor directly, not wrapped in a dict
            action, _, outputs = agent.act(state_tensor, timestep=self.step, timesteps=self.max_steps)
            action_idx = action.item() if hasattr(action, 'item') else int(action[0])
        
        return action_idx

    def _finalize_phase_skrl(self, tl, tl_id, st):
        """
        Called once, when a green expires. Records the overall
        metrics for that full green, pushes to SKRL memory, etc.
        """
        self.queue_length[tl_id] = st["queue_length"]
        self.outflow_rate[tl_id] = st["outflow"]
        self.travel_delay[tl_id] = st["travel_delay_sum"]
        self.travel_time[tl_id] = st["travel_time_sum"]
        self.waiting_time[tl_id] = st["waiting_time"]

        reward = self.get_reward(tl_id, st["phase"])
        next_state = self.get_state(tl, st["phase"])
        done = self.step >= self.max_steps

        # Store experience in SKRL memory
        if st["observation"] is not None:
            memory = self.memories[tl_id]
            
            # Convert to tensors
            state_tensor = torch.FloatTensor(st["observation"]).unsqueeze(0).to(self.device)
            action_tensor = torch.LongTensor([st["action_idx"]]).to(self.device)
            reward_tensor = torch.FloatTensor([reward]).to(self.device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            done_tensor = torch.BoolTensor([done]).to(self.device)
            
            # Add to memory
            memory.add_samples(
                states=state_tensor,
                actions=action_tensor,
                rewards=reward_tensor,
                next_states=next_state_tensor,
                terminated=done_tensor,
                truncated=done_tensor  # For SKRL compatibility
            )

        self.history["agent_reward"][tl_id].append(reward)

        # reset for next green
        for key in [
            "travel_delay_sum",
            "travel_time_sum",
            "density_sum",
            "outflow",
            "queue_length",
            "waiting_time",
        ]:
            st[key] = 0

    def _train_skrl_agents(self):
        """
        Train all SKRL DQN agents
        """
        for tl_id, agent in self.agents.items():
            memory = self.memories[tl_id]
            
            # Only train if we have enough samples
            if len(memory) >= agent.cfg["batch_size"]:
                # Train the agent
                agent.pre_interaction(timestep=self.step, timesteps=self.max_steps)
                agent.post_interaction(timestep=self.step, timesteps=self.max_steps)

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
        Turn each detector's deque of (t,count) into a veh/s over
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
        print("Training (SKRL)...")
        start_time = time.time()

        # SKRL agents are trained during simulation, so no additional training needed here
        # But we can do additional training epochs if desired
        for _ in range(min(self.epoch // 10, 100)):  # Reduced epochs since SKRL trains continuously
            self._train_skrl_agents()

        training_time = time.time() - start_time
        print("Training ended")
        print("---------------------------------------")

        # save episode plot
        self.save_plot(episode=episode)

        # reset step counter
        self.step = 0

        # Note: reset_history() moved to train.py after performance tracking
        return training_time

    def _flush_step_metrics(self, tl, tl_id, st):
        """
        Every 60 steps, average the step accumulators and append
        to self.history, then zero them out.
        """
        # Helper to compute 60-step average
        avg = lambda name: st[f"step_{name}_sum"] / 60

        # Append other metrics as before
        for metric in [
            "travel_delay",
            "travel_time",
            "density",
            "queue_length",
            "waiting_time",
        ]:
            val = avg(metric)
            self.history[metric][tl_id].append(val)
            st[f"step_{metric}_sum"] = 0

        self.history["outflow"][tl_id].append(st[f"step_outflow_sum"])
        st[f"step_outflow_sum"] = 0

    # === All the remaining methods are identical to original simulation ===
    
    def estimate_fd_params(self, tl_id, window=60, min_history=10, ema_alpha=None):
        """
        Estimate per-lane saturation_flow, critical_density, jam_density for a traffic light
        using recent flow and density history.
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

            s_new = (1 - ema_alpha) * current_s + ema_alpha * s_emp
            kc_new = (1 - ema_alpha) * current_kc + ema_alpha * k_c_emp
            kj_new = (1 - ema_alpha) * current_kj + ema_alpha * k_j_emp

            # store for next call
            self._ema_saturation_flow = s_new
            self._ema_critical_density = kc_new
            self._ema_jam_density = kj_new

            return s_new, kc_new, kj_new

        return s_emp, k_c_emp, k_j_emp

    def get_reward(self, traffic_light_id, phase):
        return (
            self.weight["outflow_rate"]
            * self.outflow_rate_normalizer.normalize(
                self.outflow_rate[traffic_light_id]
            )
            + self.weight["delay"]
            * self.travel_delay_normalizer.normalize(
                self.travel_delay[traffic_light_id]
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
                    filename=f"dqn_skrl_{self.loss_type}_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                )

            # Save metrics as DataFrame
            self.save_metrics_to_dataframe(episode=episode)

            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def save_metrics_to_dataframe(self, episode=None):
        """
        Save metrics per traffic light as pandas DataFrame.
        """
        data_records = []

        # Only collect specified system metrics
        target_metrics = [
            "density",
            "outflow",
            "queue_length",
            "travel_delay",
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
                                    "simulation_type": f"dqn_skrl_{self.loss_type}",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}dqn_skrl_{self.loss_type}_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}dqn_skrl_{self.loss_type}_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"DQN SKRL {self.loss_type} metrics DataFrame saved to {filename}")

        return df

    # === Helper methods (identical to original) ===
    
    def get_yellow_phase(self, green_phase):
        return green_phase.replace("G", "y")

    def set_yellow_phase(self, tlsId, green_phase):
        yellow_phase = self.get_yellow_phase(green_phase)
        traci.trafficlight.setPhaseDuration(tlsId, 3)
        traci.trafficlight.setRedYellowGreenState(tlsId, yellow_phase)

    def set_green_phase(self, tlsId, duration, new_phase):
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)

    def get_state(self, traffic_light, current_phase):
        """
        Get the current state at a specific traffic light in the simulation.
        """
        state_vector = []

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            # Skip phases with no movements (safety check)
            if not movements:
                continue

            waiting_time = 0
            queue_length = 0
            num_vehicles = 0

            for detector_id in movements:
                waiting_time += self.get_waiting_time(detector_id)
                queue_length += self.get_queue_length(detector_id)
                num_vehicles += self.get_num_vehicles(detector_id)

            # Append per-phase state: [waiting_time, queue_length, num_vehicles]
            state_vector.extend([waiting_time, queue_length, num_vehicles])

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

        # Append current phase
        current_phase_idx = phase_to_index(
            current_phase, self.actions_map[traffic_light["id"]], 0
        )
        state_vector.extend([current_phase_idx])

        # For SKRL, we need a fixed input dimension
        target_dim = self.agent_cfg["num_states"]
        padding = target_dim - len(state_vector) - 2  # 2 for DESRA
        state_vector.extend([0] * padding)

        # Convert phase to index
        desra_phase_idx = phase_to_index(
            best_phase, self.actions_map[traffic_light["id"]], 0
        )

        # Append DESRA guidance
        state_vector.extend([desra_phase_idx, desra_green])

        # Ensure we have the correct dimension
        if len(state_vector) != target_dim:
            # Truncate or pad as needed
            if len(state_vector) > target_dim:
                state_vector = state_vector[:target_dim]
            else:
                state_vector.extend([0] * (target_dim - len(state_vector)))

        return np.array(state_vector, dtype=np.float32)

    # === All the traffic measurement methods (identical to original) ===
    
    def get_queue_length(self, detector_id):
        length = traci.lanearea.getLength(detector_id)
        if length <= 0:
            return 0.0
        return traci.lanearea.getLastStepOccupancy(detector_id) / 100 * length

    def get_free_capacity(self, detector_id):
        lane_length = traci.lanearea.getLength(detector_id)
        occupied_length = (
            traci.lanearea.getLastStepOccupancy(detector_id) / 100 * lane_length
        )
        return (lane_length - occupied_length) / lane_length

    def get_density(self, detector_id):
        return traci.lanearea.getLastStepOccupancy(detector_id) / 100

    def get_waiting_time(self, detector_id):
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    def get_sum_travel_delay(self, traffic_light) -> float:
        delay_sum = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            speed_limit = traci.lane.getMaxSpeed(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            if speed_limit > 0:
                delay = 1.0 - (mean_speed / speed_limit)
                delay_sum += max(0.0, delay)
        return delay_sum

    def get_sum_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

    def get_sum_density(self, traffic_light):
        total_veh = 0
        total_length = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            try:
                count = traci.lane.getLastStepVehicleNumber(lane)
                lane_len = traci.lane.getLength(lane)
                total_veh += count
                total_length += lane_len
            except Exception:
                pass
        if total_length > 0:
            return total_veh / total_length
        else:
            return 0.0

    def get_num_phases(self, traffic_light):
        return len(traffic_light["phase"])

    def get_movements_from_phase(self, traffic_light, phase_str):
        phase_index = traffic_light["phase"].index(phase_str)
        active_street = str(phase_index + 1)
        active_detectors = [
            det["id"]
            for det in traffic_light["detectors"]
            if det["street"] == active_street
        ]
        return active_detectors

    def get_sum_queue_length(self, traffic_light):
        queue_lengths = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            length = traci.lane.getLength(lane)
            if length <= 0:
                queue_lengths.append(0.0)
            else:
                queue_lengths.append(
                    traci.lane.getLastStepOccupancy(lane) / 100 * length
                )
        return np.sum(queue_lengths) if queue_lengths else 0.0

    def get_sum_waiting_time(self, traffic_light):
        total_waiting_time = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            total_waiting_time += traci.lane.getWaitingTime(lane)
        return total_waiting_time

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []

    def get_num_vehicles(self, detector_id):
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def save_models(self, episode):
        """Save SKRL models"""
        for tl_id, agent in self.agents.items():
            model_path = f"{self.path}skrl_dqn_{tl_id}_episode_{episode}.pt"
            agent.save(model_path)
            print(f"SKRL DQN model saved for {tl_id}: {model_path}")

    def load_models(self, episode):
        """Load SKRL models"""
        for tl_id, agent in self.agents.items():
            model_path = f"{self.path}skrl_dqn_{tl_id}_episode_{episode}.pt"
            try:
                agent.load(model_path)
                print(f"SKRL DQN model loaded for {tl_id}: {model_path}")
            except FileNotFoundError:
                print(f"No saved model found for {tl_id}: {model_path}")
