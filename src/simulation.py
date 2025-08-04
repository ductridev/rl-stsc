"""
Main simulation class using SKRL for reinforcement learning.
Refactored to use separate modules for better organization.
"""

import numpy as np
import torch
import time
import traci
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from visualization import Visualization
from normalizer import Normalizer
from desra import DESRA
from sumo import SUMO
from accident_manager import AccidentManager
from vehicle_tracker import VehicleTracker
from agents.skrl_agent_manager import SKRLAgentManager
from sim_utils.phase_manager import index_to_action, phase_to_index
from sim_utils.traffic_metrics import TrafficMetrics
from comparison_utils import SimulationComparison


class Simulation(SUMO):
    """Traffic light simulation using SKRL for reinforcement learning"""

    def __init__(
        self,
        visualization: Visualization,
        agent_cfg: Dict,
        max_steps: int,
        traffic_lights: List[Dict],
        accident_manager: AccidentManager,
        interphase_duration: int = 3,
        epoch: int = 1000,
        path: str = None,
        training_steps: int = 300,
        updating_target_network_steps: int = 100,
        save_interval: int = 2,
    ):
        # Initialize parent class
        super().__init__()

        # Core configuration
        self.visualization = visualization
        self.agent_cfg = agent_cfg
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.interphase_duration = interphase_duration
        self.epoch = epoch
        self.path = path
        self.training_steps = training_steps
        self.updating_target_network_steps = updating_target_network_steps
        self.save_interval = save_interval

        # Normalizers
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_delay_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()

        # Vehicle tracking
        self.vehicle_tracker = VehicleTracker(path=self.path)

        # Simulation state
        self.step = 0
        self.num_actions = {}
        self.actions_map = {}

        # Agent states and memories
        self.agent_reward = {}
        self.agent_state = {}
        self.agent_old_state = {}

        # Traffic metrics
        self.arrival_buffers = defaultdict(lambda: deque())
        self.last_desra_time = {}
        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_delay = {}
        self.travel_time = {}
        self.waiting_time = {}
        self.phase = {}

        # History tracking
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

        # Initialize state
        self.initState()

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Setup SKRL agent manager
        self.agent_manager = SKRLAgentManager(
            simulation_instance=self,
            agent_cfg=self.agent_cfg,
            traffic_lights=self.traffic_lights,
            updating_target_network_steps=self.updating_target_network_steps,
            device=self.device,
        )

        # DESRA setup
        self.desra = DESRA(interphase_duration=self.interphase_duration)

        # DESRA configuration (default to detector-specific parameters)
        self.desra_use_global_params = False
        self.desra_global_saturation_flow = None
        self.desra_global_critical_density = None
        self.desra_global_jam_density = None

        # Comparison utility
        self.comparison = SimulationComparison(path=self.path) if self.path else None

        # Traffic light states
        self.tl_states = {}

        # For backward compatibility with training script
        self.agent = self  # Point to self so training script can access methods

    @property
    def agents(self):
        """Property to access SKRL agents for compatibility"""
        return self.agent_manager.agents

    @property
    def memories(self):
        """Property to access SKRL memories for compatibility"""
        return self.agent_manager.memories

    @property
    def models(self):
        """Property to access SKRL models for compatibility"""
        return self.agent_manager.models

    def save(self, path: str):
        """Save all SKRL models to a path"""
        self.agent_manager.save_models(path)

    def save_checkpoint(self, path: str, episode: int = None, epsilon: float = None):
        """Save all SKRL model checkpoints"""
        self.agent_manager.save_checkpoints(path, episode, epsilon)

    def initState(self):
        """Initialize simulation state for all traffic lights"""
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]

            # Initialize the number of actions
            self.num_actions[traffic_light_id] = len(traffic_light["phase"])

            # Initialize the action map
            self.actions_map[traffic_light_id] = {}
            for i, phase in enumerate(traffic_light["phase"]):
                self.actions_map[traffic_light_id][i] = {"phase": phase}

            # Initialize metrics
            self.agent_reward[traffic_light_id] = 0
            self.agent_state[traffic_light_id] = 0
            self.agent_old_state[traffic_light_id] = 0
            self.queue_length[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.travel_delay[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0
            self.phase[traffic_light_id] = None

            # Initialize history
            for key in self.history:
                self.history[key][traffic_light_id] = []

            # Initialize arrival buffers
            for phase in traffic_light["phase"]:
                for det in self.get_movements_from_phase(traffic_light, phase):
                    _ = self.arrival_buffers[det]

    def select_action(
        self, tl_id: str, state: np.ndarray, epsilon: float
    ) -> Tuple[float, int]:
        """Select action using SKRL agent manager"""
        return self.agent_manager.select_action(tl_id, state, epsilon)

    def store_transition(
        self,
        tl_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """Store transition in agent's memory"""
        self.agent_manager.store_transition(
            tl_id, state, action, reward, next_state, done
        )

    def train_agents(self):
        """Train all agents using SKRL"""
        return self.agent_manager.train_agents(self.step, self.max_steps)

    def run(self, epsilon: float, episode: int):
        """Run simulation with SKRL agents"""
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        # Build detector list
        self.all_detectors = [
            det
            for tl in self.traffic_lights
            for phase_str in tl["phase"]
            for det in self.get_movements_from_phase(tl, phase_str)
        ]

        # Initialize DESRA timing
        for det in self.all_detectors:
            self.last_desra_time[det] = traci.simulation.getTime()

        start_time = time.time()

        # Initialize traffic light states
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            self.tl_states[tl_id] = {
                "green_time_remaining": 10,
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
                "green_time": 10,
                "step_travel_delay_sum": 0,
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
            # Action selection for each traffic light
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                state = self.tl_states[tl_id]

                # Handle yellow phase countdown
                if state["yellow_time_remaining"] > 0:
                    state["yellow_time_remaining"] -= 1
                    continue

                # Skip if green phase is still running
                if state["green_time_remaining"] > 0:
                    continue

                # Handle interphase (yellow) transition
                if not state["interphase"] and state["old_phase"] != state["phase"]:
                    self.set_yellow_phase(tl_id, state["phase"])
                    state["yellow_time_remaining"] = self.interphase_duration
                    state["interphase"] = True
                    continue

                # Green phase ended - select new action
                state["interphase"] = False

                # Get current state
                current_state = self.get_state(tl, state["phase"])

                # Select action using SKRL agent
                random_val, action_idx = self.select_action(
                    tl_id, current_state, epsilon
                )

                # Convert action to phase
                new_phase = index_to_action(action_idx, self.actions_map[tl_id])

                # Calculate green time
                green_time = max(
                    1, min(state["green_time"], self.max_steps - self.step)
                )

                # Apply new phase
                self.set_green_phase(tl_id, green_time, new_phase)

                # Update state
                state.update(
                    {
                        "phase": new_phase,
                        "old_phase": state["phase"],
                        "green_time": green_time,
                        "green_time_remaining": green_time,
                        "old_vehicle_ids": TrafficMetrics.get_vehicles_in_phase(
                            tl, new_phase
                        ),
                        "state": current_state,
                        "action_idx": action_idx,
                    }
                )

            # Simulation step
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            num_vehicles_out += traci.simulation.getArrivedNumber()
            self.step += 1

            # Update vehicle tracking
            self.vehicle_tracker.update_stats(self.step)

            # Update DESRA traffic parameters for real-time adaptation
            current_time = traci.simulation.getTime()
            for det in self.all_detectors:
                self.desra.update_traffic_parameters(det, current_time)

            # Print stats periodically
            if self.step % 1000 == 0:
                current_stats = self.vehicle_tracker.get_current_stats()
                print(
                    f"Step {self.step}: Running={current_stats['total_running']}, "
                    f"Departed={current_stats['total_departed']}, "
                    f"Arrived={current_stats['total_arrived']}"
                )

            self._record_arrivals()

            # Collect metrics and handle phase transitions
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                st = self.tl_states[tl_id]
                phase = st["phase"]

                # Calculate metrics
                new_ids = TrafficMetrics.get_vehicles_in_phase(tl, phase)
                outflow = sum(1 for vid in st["old_vehicle_ids"] if vid not in new_ids)
                st["old_vehicle_ids"] = new_ids

                sum_travel_delay = TrafficMetrics.get_sum_travel_delay(tl)
                sum_travel_time = TrafficMetrics.get_sum_travel_time(tl)
                sum_density = TrafficMetrics.get_sum_density(tl)
                sum_queue_length = TrafficMetrics.get_sum_queue_length(tl)
                sum_waiting_time = TrafficMetrics.get_sum_waiting_time(tl)

                # Update metrics
                st["step_outflow_sum"] += outflow
                st["step_travel_delay_sum"] += sum_travel_delay
                st["step_travel_time_sum"] = sum_travel_time
                st["step_density_sum"] += sum_density
                st["step_queue_length_sum"] += sum_queue_length
                st["step_waiting_time_sum"] = sum_waiting_time

                st["outflow"] += outflow
                st["travel_delay_sum"] = sum_travel_delay
                st["travel_time_sum"] = sum_travel_time
                st["queue_length"] = sum_queue_length
                st["waiting_time"] = sum_waiting_time

                # Handle green time countdown
                if st["green_time_remaining"] > 0:
                    st["green_time_remaining"] -= 1
                else:
                    # Phase ended - finalize and store experience
                    self._finalize_phase_skrl(tl, tl_id, st)

                # Periodic metric flushing
                if self.step > 0 and self.step % 60 == 0:
                    self._flush_step_metrics(tl, tl_id, st)

            # Training
            if self.step > 0 and self.step % self.training_steps == 0:
                print(f"Training per {self.training_steps} steps...")
                train_start = time.time()

                for _ in range(self.epoch):
                    loss = self.train_agents()

                print(f"Training took {time.time() - train_start:.2f}s")

        traci.close()
        sim_time = time.time() - start_time
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} arrived."
        )

        # Calculate total reward
        total_reward = sum(
            np.sum(self.history["agent_reward"][tl["id"]]) for tl in self.traffic_lights
        )
        print(f"Total reward: {total_reward} - Epsilon: {epsilon}")

        # Save vehicle statistics
        self.vehicle_tracker.print_summary("skrl_dqn")
        self.vehicle_tracker.save_logs(episode, "skrl_dqn")
        self.vehicle_tracker.reset()

        # Print DESRA parameter summary
        self.print_desra_summary()

        return sim_time, self._finalize_episode(epsilon, episode)

    def _finalize_phase_skrl(self, tl: Dict, tl_id: str, st: Dict):
        """Finalize phase and store experience in SKRL memory"""
        # Calculate reward
        reward = self.get_reward(tl_id, st["phase"])

        # Get next state
        next_state = self.get_state(tl, st["phase"])

        # Check if done
        done = self.step >= self.max_steps

        # Store transition in SKRL memory
        if st["state"] is not None:
            self.store_transition(
                tl_id=tl_id,
                state=st["state"],
                action=st["action_idx"],
                reward=reward,
                next_state=next_state,
                done=done,
            )

        # Update history
        self.history["agent_reward"][tl_id].append(reward)

        # Reset phase metrics
        for key in [
            "travel_delay_sum",
            "travel_time_sum",
            "outflow",
            "queue_length",
            "waiting_time",
        ]:
            st[key] = 0

    def _record_arrivals(self):
        """Record vehicle arrivals for DESRA"""
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

    def _flush_step_metrics(self, tl: Dict, tl_id: str, st: Dict):
        """Flush step metrics to history"""
        # Calculate averages
        avg = lambda name: st[f"step_{name}_sum"] / 60

        # Append to history
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

        self.history["outflow"][tl_id].append(st["step_outflow_sum"])
        st["step_outflow_sum"] = 0

    def _finalize_episode(self, epsilon: float, episode: int):
        """Finalize episode and handle plotting/saving"""
        print("Training completed")
        print("---------------------------------------")

        # Update target networks periodically
        if episode % self.save_interval == 0 and episode > 0:
            self.agent_manager.update_target_networks(self.step, self.max_steps)

        # Save plots
        self.save_plot(episode=episode)

        # Reset for next episode
        self.step = 0
        self.reset_history()

        return 0.0  # Return dummy training time

    def reset_history(self):
        """Reset history for next episode"""
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []

    def save_model(self, episode: int):
        """Save SKRL models"""
        if self.path:
            self.agent_manager.save_models(self.path, episode)
            print(f"Saved models at episode {episode}")

    def load_model(self, episode: int):
        """Load SKRL models"""
        if self.path:
            self.agent_manager.load_models(self.path, episode)

    def get_state(self, tl: Dict, phase: str) -> np.ndarray:
        """Get state representation for a traffic light phase"""
        state_vector = []

        for phase_str in tl["phase"]:
            movements = self.get_movements_from_phase(tl, phase_str)

            # Skip phases with no movements (safety check)
            if not movements:
                continue

            waiting_time = 0
            queue_length = 0
            num_vehicles = 0

            for detector_id in movements:
                waiting_time += TrafficMetrics.get_waiting_time(detector_id)
                queue_length += TrafficMetrics.get_queue_length(detector_id)
                num_vehicles += TrafficMetrics.get_num_vehicles(detector_id)

            # Append per-phase state: [waiting_time, queue_length, num_vehicles]
            state_vector.extend([waiting_time, queue_length, num_vehicles])

        # Compute per-detector q_arr over [last_call, now]
        q_arr_dict = self._compute_arrival_flows()

        # Get current simulation time for DESRA parameter updates
        current_time = traci.simulation.getTime()

        # Get DESRA configuration for this traffic light
        (
            use_global_params,
            global_saturation_flow,
            global_critical_density,
            global_jam_density,
        ) = self.get_desra_parameters(tl["id"])

        # DESRA recommended phase and green time with real-time parameter updates
        best_phase, desra_green = self.desra.select_phase_with_desra_hints(
            tl,
            q_arr_dict,
            current_time=current_time,
            use_global_params=use_global_params,
            global_saturation_flow=global_saturation_flow,
            global_critical_density=global_critical_density,
            global_jam_density=global_jam_density,
        )

        # Append current phase
        current_phase_idx = phase_to_index(phase, self.actions_map[tl["id"]], 0)
        state_vector.extend([current_phase_idx])

        padding = (
            self.agent_cfg["num_states"] - len(state_vector) - 1
        )  # 1 for DESRA phase instead of green time
        state_vector.extend([0] * padding)

        # Convert phase to index
        desra_phase_idx = phase_to_index(best_phase, self.actions_map[tl["id"]], 0)

        # Append DESRA guidance (using DESRA phase instead of green time)
        state_vector.extend([desra_phase_idx])

        assert (
            len(state_vector) == self.agent_cfg["num_states"]
        ), f"State vector length {len(state_vector)} does not match input_dim {self.agent_cfg['num_states']}"

        return np.array(state_vector, dtype=np.float32)

    def get_reward(self, tl_id: str, phase: str) -> float:
        """Calculate reward for a traffic light action"""
        weight = self.agent_cfg.get(
            "weight",
            {
                "outflow_rate": 1.0,
                "delay": -1.0,
                "waiting_time": -1.0,
                "switch_phase": -0.1,
                "travel_time": -1.0,
                "queue_length": -1.0,
            },
        )

        return (
            weight["outflow_rate"]
            * self.outflow_rate_normalizer.normalize(self.outflow_rate[tl_id])
            + weight["delay"]
            * self.travel_delay_normalizer.normalize(self.travel_delay[tl_id])
            + weight["waiting_time"]
            * self.waiting_time_normalizer.normalize(self.waiting_time[tl_id])
            + weight["switch_phase"] * (int)(self.phase[tl_id] != phase)
            + weight["travel_time"]
            * self.travel_time_normalizer.normalize(self.travel_time[tl_id])
            + weight["queue_length"]
            * self.queue_length_normalizer.normalize(self.queue_length[tl_id])
        )

    def get_movements_from_phase(self, tl: Dict, phase_str: str) -> List[str]:
        """Get movement detectors from a phase"""
        return TrafficMetrics.get_movements_from_phase(tl, phase_str)

    def get_yellow_phase(self, green_phase: str) -> str:
        """Convert green phase to yellow phase"""
        return green_phase.replace("G", "y")

    def set_yellow_phase(self, tl_id: str, green_phase: str):
        """Set yellow phase for a traffic light"""
        yellow_phase = self.get_yellow_phase(green_phase)
        traci.trafficlight.setPhaseDuration(tl_id, 3)
        traci.trafficlight.setRedYellowGreenState(tl_id, yellow_phase)

    def set_green_phase(self, tl_id: str, green_time: int, phase: str):
        """Set green phase for a traffic light"""
        traci.trafficlight.setPhaseDuration(tl_id, green_time)
        traci.trafficlight.setRedYellowGreenState(tl_id, phase)

    def _compute_arrival_flows(self):
        """Turn each detector's deque of (t,count) into a veh/s over
        the exact interval since the last DESRA call."""
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

    def estimate_fd_params(self, tl_id, window=60, min_history=10, ema_alpha=None):
        """
        Legacy method for estimating fundamental diagram parameters.

        NOTE: This method is kept for backward compatibility. The new DESRA class
        now handles real-time parameter estimation internally using detector-specific
        traffic measurements. Use DESRA.get_detector_parameters() instead.

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

    def save_plot(self, episode: int):
        """Save episode plots"""
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
                    filename=f"skrl_dqn_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                )

            # Save metrics as DataFrame
            self.save_metrics_to_dataframe(episode=episode)

            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def save_metrics_to_dataframe(self, episode=None):
        """Save metrics per traffic light as pandas DataFrame"""
        import pandas as pd

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
                                    "simulation_type": "skrl_dqn",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}skrl_dqn_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}skrl_dqn_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"SKRL DQN metrics DataFrame saved to {filename}")

        return df

    def configure_desra_parameters(
        self,
        use_global_params=False,
        global_saturation_flow=None,
        global_critical_density=None,
        global_jam_density=None,
    ):
        """
        Configure DESRA to use either global or detector-specific parameters.

        Args:
            use_global_params: If True, use global parameters for all detectors
            global_saturation_flow: Global saturation flow rate (veh/s)
            global_critical_density: Global critical density (veh/m)
            global_jam_density: Global jam density (veh/m)
        """
        self.desra_use_global_params = use_global_params
        self.desra_global_saturation_flow = global_saturation_flow
        self.desra_global_critical_density = global_critical_density
        self.desra_global_jam_density = global_jam_density

        print(f"DESRA configured: use_global_params={use_global_params}")
        if use_global_params:
            print(
                f"Global parameters: s={global_saturation_flow}, "
                f"kc={global_critical_density}, kj={global_jam_density}"
            )

    def get_desra_parameters(self, tl_id):
        """
        Get DESRA parameters for a traffic light, either global or detector-specific.

        Args:
            tl_id: Traffic light ID

        Returns:
            Tuple of (use_global_params, global_saturation_flow, global_critical_density, global_jam_density)
        """
        if hasattr(self, "desra_use_global_params") and self.desra_use_global_params:
            return (
                True,
                getattr(self, "desra_global_saturation_flow", None),
                getattr(self, "desra_global_critical_density", None),
                getattr(self, "desra_global_jam_density", None),
            )
        else:
            return (False, None, None, None)

    def get_desra_statistics(self):
        """
        Get current DESRA parameter estimates for all detectors.

        Returns:
            Dict with detector statistics including estimated parameters and buffer sizes
        """
        stats = {}
        for det in self.all_detectors:
            params = self.desra.get_detector_parameters(det)
            stats[det] = {
                "estimated_saturation_flow": params["saturation_flow"],
                "estimated_critical_density": params["critical_density"],
                "estimated_jam_density": params["jam_density"],
                "vehicle_count_buffer_size": len(
                    self.desra.vehicle_counts.get(det, [])
                ),
                "occupancy_buffer_size": len(self.desra.occupancies.get(det, [])),
                "flow_rate_buffer_size": len(self.desra.flow_rates.get(det, [])),
                "density_buffer_size": len(self.desra.densities.get(det, [])),
            }
        return stats

    def print_desra_summary(self):
        """Print a summary of DESRA parameter estimates."""
        print("\n=== DESRA Parameter Summary ===")
        print(
            f"Mode: {'Global Parameters' if self.desra_use_global_params else 'Detector-Specific Parameters'}"
        )

        if self.desra_use_global_params:
            print(f"Global Saturation Flow: {self.desra_global_saturation_flow}")
            print(f"Global Critical Density: {self.desra_global_critical_density}")
            print(f"Global Jam Density: {self.desra_global_jam_density}")
        else:
            stats = self.get_desra_statistics()
            print(f"Number of detectors: {len(stats)}")

            if stats:
                # Calculate averages
                avg_saturation_flow = np.mean(
                    [s["estimated_saturation_flow"] for s in stats.values()]
                )
                avg_critical_density = np.mean(
                    [s["estimated_critical_density"] for s in stats.values()]
                )
                avg_jam_density = np.mean(
                    [s["estimated_jam_density"] for s in stats.values()]
                )
                avg_buffer_size = np.mean(
                    [s["vehicle_count_buffer_size"] for s in stats.values()]
                )

                print(f"Average Saturation Flow: {avg_saturation_flow:.3f} veh/s")
                print(f"Average Critical Density: {avg_critical_density:.3f} veh/m")
                print(f"Average Jam Density: {avg_jam_density:.3f} veh/m")
                print(f"Average Buffer Size: {avg_buffer_size:.1f} measurements")
        print("==============================\n")
