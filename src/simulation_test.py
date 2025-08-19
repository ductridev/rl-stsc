"""
Main simulation class using SKRL for reinforcement learning.
Refactored to use separate modules for better organization.
"""

import math
import numpy as np
import torch
import time
import libsumo as traci
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple
import sys
import os
import gymnasium as gym

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from visualization import Visualization
from normalizer import Normalizer
from desra import DESRA
from .sumo import SUMO
from accident_manager import AccidentManager
from vehicle_tracker import VehicleTracker
from models.q_network import QNetwork
from agents.skrl_agent_manager import SKRLAgentManager
from sim_utils.phase_manager import index_to_action, phase_to_index
from sim_utils.traffic_metrics import TrafficMetrics, VehicleCompletionTracker, JunctionVehicleTracker
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
        model_file_path: str,
        interphase_duration: int = 3,
        path: str = None,
        use_skrl: bool = False,
        evaluation_mode: bool = True,
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
        self.path = path
        self.model_file_path = model_file_path
        self.green_time = 10
        self.use_skrl = use_skrl
        self.evaluation_mode = evaluation_mode

        # Normalizers
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.num_vehicles_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_delay_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()

        # Vehicle tracking
        self.vehicle_tracker = VehicleTracker(path=self.path)

        # Vehicle completion tracking for episode-end travel time analysis
        self.completion_tracker = VehicleCompletionTracker()

        # Junction vehicle tracking for throughput analysis
        self.junction_tracker = JunctionVehicleTracker()

        # Simulation state
        self.step = 0
        self.global_step = 0
        self.num_actions = {}
        self.actions_map = {}

        # Agent states and memories
        self.reward = {}
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
            "reward": {},
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
            "desra_usage": {},  # Track DESRA vs DQN action selections
            "completed_travel_time": {},  # New: track average completed vehicle travel times
            "junction_throughput": {},  # New: track vehicles entering junctions
            "stopped_vehicles": {},  # New: track number of stopped vehicles
        }

        # DESRA usage tracking
        self.desra_action_count = {}
        self.dqn_action_count = {}

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

        # DESRA usage tracking
        self.desra_action_count = {}
        self.dqn_action_count = {}

        # For backward compatibility with training script
        self.agent = self  # Point to self so training script can access methods

        # Decision counter for tracking training steps
        self.decision_counter = 0  # counts phase decisions per episode

        self.model = {}

        # Initialize state
        self.initState()

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
            self.reward[traffic_light_id] = 0
            self.agent_state[traffic_light_id] = 0
            self.agent_old_state[traffic_light_id] = 0
            self.queue_length[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.travel_delay[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0
            self.phase[traffic_light_id] = None

            # Initialize DESRA tracking
            self.desra_action_count[traffic_light_id] = 0
            self.dqn_action_count[traffic_light_id] = 0

            # Initialize history
            for key in self.history:
                self.history[key][traffic_light_id] = []

            # Initialize arrival buffers
            for phase in traffic_light["phase"]:
                for det in self.get_movements_from_phase(traffic_light, phase):
                    _ = self.arrival_buffers[det]

            # Initialize SKRL agent manager if enabled
            if self.use_skrl:
                self.skrl_agent_manager = SKRLAgentManager(
                    simulation_instance=self,
                    agent_cfg=self.agent_cfg,
                    traffic_lights=self.traffic_lights,
                    updating_target_network_steps=100,
                    device=self.device,
                    evaluation_mode=self.evaluation_mode
                )
                
                # Load models if path is provided
                if self.model_file_path and os.path.exists(self.model_file_path):
                    self.skrl_agent_manager.load_models(self.model_file_path, episode=1)
                    print("SKRL models loaded successfully")
            else:
                self.skrl_agent_manager = None
                self.model[traffic_light_id] = torch.load(self.model_file_path, map_location=self.device)

    def select_action(
        self, tl_id: str, state: np.ndarray, desra_phase_idx: int
    ) -> int:
        """Select action using hybrid DESRA-DQN approach"""

        assert (
            len(state) == self.agent_cfg["num_states"]
        ), f"State vector length {len(state)} does not match input_dim {self.agent_cfg['num_states']}"

        # Use SKRL agent manager if available
        if self.use_skrl and self.skrl_agent_manager:
            return self.skrl_agent_manager.select_action(
                tl_id=tl_id,
                state=state,
                step=self.step,
                max_steps=self.max_steps,
                desra_phase_idx=desra_phase_idx
            )
        
        # Fallback to original model-based approach
        state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device)

        q_values, log, adds = self.model[tl_id].act({"states": state_tensor}, role="q_network")

        predicted_actions = torch.argmax(q_values.cpu())

        print(f"Traffic Light {tl_id}: {state} - Selected action index: {q_values} - Argmax: {predicted_actions}")

        return int(predicted_actions)

    def run(self, episode: int):
        """Run simulation with SKRL agents"""
        print("Simulation started")
        print("TESTING MODE: Training disabled, epsilon=0 (pure exploitation), DESRA disabled (pure DQN)")
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
                "green_time_remaining": self.green_time,
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
                "green_time": self.green_time,
                "step_travel_delay_sum": 0,
                "step_travel_time_sum": 0,
                "step_outflow_sum": 0,
                "step_density_sum": 0,
                "step_queue_length_sum": 0,
                "step_waiting_time_sum": 0,
                "step_junction_throughput_sum": 0,
                "step_stopped_vehicles_sum": 0,
                "step_old_vehicle_ids": [],
                "step_reward_sum": 0
            }

        num_vehicles = 0
        num_vehicles_out = 0

        # Warm up 50 steps
        for _ in range(50):
            traci.simulationStep()

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
                current_state, desra_phase_idx = self.get_state(tl, state["phase"])

                # Select action using SKRL agent
                action_idx = self.select_action(
                    tl_id, current_state, desra_phase_idx
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
                        "old_vehicle_ids": TrafficMetrics.get_vehicles_in_phase(
                            tl, new_phase
                        ),
                        "green_time_remaining": green_time,
                        "state": current_state,
                        "action_idx": action_idx,
                    }
                )
                # Increment decision counter once per action selection
                self.decision_counter += 1

            # Simulation step with error handling
            if self.accident_manager:
                self.accident_manager.create_accident(current_step=self.step)
            try:
                traci.simulationStep()
                num_vehicles += traci.simulation.getDepartedNumber()
                num_vehicles_out += traci.simulation.getArrivedNumber()
            except Exception as e:
                print(f"SUMO simulation error at step {self.step}: {e}")
                # Try to handle common SUMO errors
                if "has no valid route" in str(e):
                    print("Route validation error detected - attempting to continue...")
                    # Remove problematic vehicles and continue
                    try:
                        # Get list of vehicles and remove any with route issues
                        vehicle_ids = traci.vehicle.getIDList()
                        for vid in vehicle_ids:
                            try:
                                route = traci.vehicle.getRoute(vid)
                                if not route or len(route) == 0:
                                    print(f"Removing vehicle {vid} with invalid route")
                                    traci.vehicle.remove(vid)
                            except:
                                print(f"Removing problematic vehicle {vid}")
                                try:
                                    traci.vehicle.remove(vid)
                                except:
                                    pass  # Vehicle might already be removed
                        # Try simulation step again
                        traci.simulationStep()
                        num_vehicles += traci.simulation.getDepartedNumber()
                        num_vehicles_out += traci.simulation.getArrivedNumber()
                    except Exception as e2:
                        print(f"Failed to recover from SUMO error: {e2}")
                        print("Ending episode early due to simulation error")
                        break
                else:
                    print(f"Unhandled SUMO error: {e}")
                    print("Ending episode early due to simulation error")
                    break
            self.step += 1
            self.global_step += 1
            # Removed per-step update_step calls (no direct _update usage)

            # Update vehicle tracking
            self.vehicle_tracker.update_stats(self.step)

            # Track completed vehicles for travel time analysis
            self.completion_tracker.update_completed_vehicles()

            # Update junction throughput tracking
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                st = self.tl_states[tl_id]
                # Assume junction_id is in the traffic light config, or derive from tl_id
                junction_id = tl.get("junction_id", tl_id)  # Use junction_id if available, else use tl_id
                new_count, new_vehicles = self.junction_tracker.update_junction(junction_id)
                
                # Accumulate throughput for this step instead of directly appending
                if new_count > 0:
                    st["step_junction_throughput_sum"] += new_count

            # Update DESRA traffic parameters for real-time adaptation
            # Only update every 100 steps to reduce computational overhead
            if self.step % 100 == 0:
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
                current_phase = st["phase"]

                # Calculate metrics
                new_ids = TrafficMetrics.get_vehicles_in_phase(tl, current_phase)
                outflow = sum(1 for vid in st["old_vehicle_ids"] if vid not in new_ids)
                st["old_vehicle_ids"] = new_ids

                sum_travel_delay = TrafficMetrics.get_sum_travel_delay(tl)
                sum_travel_time = TrafficMetrics.get_sum_travel_time(tl)
                sum_density = TrafficMetrics.get_sum_density(tl)
                sum_queue_length = TrafficMetrics.get_sum_queue_length(tl)
                sum_waiting_time = TrafficMetrics.get_sum_waiting_time(tl)
                mean_waiting_time = TrafficMetrics.get_mean_waiting_time(tl)
                stopped_vehicles_count = TrafficMetrics.count_stopped_vehicles_for_traffic_light(tl)

                # Update metrics
                st["step_outflow_sum"] += outflow
                st["step_travel_delay_sum"] += sum_travel_delay
                st["step_travel_time_sum"] += sum_travel_time
                st["step_density_sum"] += sum_density
                st["step_queue_length_sum"] += sum_queue_length
                st["step_waiting_time_sum"] += mean_waiting_time
                st["step_stopped_vehicles_sum"] += stopped_vehicles_count

                st["outflow"] += outflow
                st["travel_delay_sum"] = sum_travel_delay
                st["travel_time_sum"] = sum_travel_time
                st["queue_length"] = sum_queue_length
                st["waiting_time"] = mean_waiting_time

                # Handle green time countdown
                if st["green_time_remaining"] > 1:
                    st["green_time_remaining"] -= 1
                else:
                    # Phase ended - finalize and store experience
                    self._finalize_phase_skrl(tl, tl_id, st)
                    st["green_time_remaining"] -= 1

                # Periodic metric flushing
                if self.step > 0 and self.step % 300 == 0:
                    self._flush_step_metrics(tl, tl_id, st)

                # FIX: Removed redundant step-based training to avoid double training

        traci.close()
        sim_time = time.time() - start_time
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} arrived."
        )

        # Save vehicle statistics
        self.vehicle_tracker.print_summary("skrl_dqn")
        self.vehicle_tracker.save_logs(episode, "skrl_dqn")
        # self.vehicle_tracker.reset()

        # Print DESRA parameter summary
        self.print_desra_summary()

        return sim_time, self._finalize_episode(episode)

    def _finalize_phase_skrl(self, tl: Dict, tl_id: str, st: Dict):
        """Finalize phase and store experience in SKRL memory"""
        self.queue_length[tl_id] = st["queue_length"]
        self.outflow_rate[tl_id] = st["outflow"]
        self.travel_delay[tl_id] = st["travel_delay_sum"]
        self.travel_time[tl_id] = st["travel_time_sum"]
        self.waiting_time[tl_id] = st["waiting_time"]

        # Calculate reward
        reward = self.get_reward(tl_id, st["phase"])

        # Always record reward when phase ends (not just every 60 steps)
        st["step_reward_sum"] += reward

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
        avg = lambda name: st[f"step_{name}_sum"] / 300

        # Append to history
        for metric in [
            "travel_delay",
            "travel_time",
            "density",
            "queue_length",
            "waiting_time",
            "stopped_vehicles",
            "reward"
        ]:
            val = avg(metric)
            self.history[metric][tl_id].append(val)
            st[f"step_{metric}_sum"] = 0

        self.history["outflow"][tl_id].append(st["step_outflow_sum"])
        st["step_outflow_sum"] = 0
        
        # Add junction throughput to history (sum over 300 steps)
        if tl_id not in self.history["junction_throughput"]:
            self.history["junction_throughput"][tl_id] = []
        self.history["junction_throughput"][tl_id].append(st["step_junction_throughput_sum"])
        st["step_junction_throughput_sum"] = 0

    def _finalize_episode(self, episode: int):
        """Finalize episode and handle plotting/saving"""
        print("Testing completed")
        print("---------------------------------------")

        # Record episode-end completed vehicle travel time for all traffic lights
        episode_avg_travel_time = self.completion_tracker.get_average_total_travel_time()
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            if tl_id not in self.history["completed_travel_time"]:
                self.history["completed_travel_time"][tl_id] = []
            self.history["completed_travel_time"][tl_id].append(episode_avg_travel_time)
        
        print(f"Episode {episode}: {self.completion_tracker.get_completed_count()} vehicles completed, "
              f"average travel time: {episode_avg_travel_time:.2f}s")

        # Save plots
        self.save_plot(episode=episode)
        
        # Print DESRA usage summary
        self.print_desra_usage_summary()

        # Reset step counter for next episode
        self.step = 0
        
        # Clear arrival buffers to prevent memory leak
        self.clear_arrival_buffers()
        
        # Clear DESRA buffers to prevent memory accumulation
        if hasattr(self.desra, 'clear_buffers'):
            self.desra.clear_buffers()

        # Reset decision counter
        self.decision_counter = 0  # reset decision counter

        # Reset DESRA tracking for next episode
        self.reset_desra_tracking()

        # Reset junction tracking for next episode
        self.junction_tracker.reset_all()

        # Reset vehicle stop tracking for next episode
        TrafficMetrics.reset_vehicle_stop_tracker()

        return 0.0  # Return dummy training time

    def reset_history(self):
        """Reset history for next episode"""
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []

    def clear_arrival_buffers(self):
        """Clear arrival buffers to prevent memory accumulation"""
        for det in self.arrival_buffers:
            self.arrival_buffers[det].clear()
        print("Cleared arrival buffers to prevent memory leaks")

    def get_state(self, tl: Dict, phase: str) -> np.ndarray:
        """Get state representation for a traffic light phase"""
        state_vector = []

        for phase_idx, phase_str in enumerate(tl["phase"]):
            movements = self.get_movements_from_phase(tl, phase_str)

            num_vehicles = 0
            highest_queue_length = 0

            lane_waiting_time = 0.0
            for det in movements:
                for veh in traci.lanearea.getLastStepVehicleIDs(det):
                    # Get cumulative waiting time for all vehicles in lane
                    lane_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh)
                # Get number of vehicles currently in lane
                num_vehicles += traci.lanearea.getLastStepVehicleNumber(det)

                queue_length = TrafficMetrics.get_queue_length(det)
                highest_queue_length = queue_length if queue_length > highest_queue_length else highest_queue_length

            # Fill per-phase features
            state_vector.append([
                lane_waiting_time / num_vehicles if num_vehicles > 0 else 0,
                num_vehicles,
                highest_queue_length,
            ])

        q_arr_dict = self._compute_arrival_flows()
        current_time = traci.simulation.getTime()

        (
            use_global_params,
            global_saturation_flow,
            global_critical_density,
            global_jam_density,
        ) = self.get_desra_parameters(tl["id"])

        best_phase, desra_green = self.desra.select_phase_with_desra_hints(
            tl,
            q_arr_dict,
            current_time=current_time,
            use_global_params=use_global_params,
            global_saturation_flow=global_saturation_flow,
            global_critical_density=global_critical_density,
            global_jam_density=global_jam_density,
        )

        # Convert DESRA phase to index
        desra_phase_idx = phase_to_index(best_phase, self.actions_map[tl["id"]], 0)

        # Check if desra does not select empty phase follow the state vector
        best_phase_idx = self.choose_best_phase(state_vector, desra_phase_idx)

        for i in range(len(state_vector)):
            is_desra = 1.0 if i == best_phase_idx else 0.0
            state_vector[i].append(is_desra)

        flat_state = [f for phase_features in state_vector for f in phase_features]

        # Current phase to index
        current_phase_idx = phase_to_index(phase, self.actions_map[tl["id"]], 0)

        # flat_state.extend([current_phase_idx, best_phase_idx])

        # assert (
        #     len(state_vector) == self.agent_cfg["num_states"]
        # ), f"State vector length {len(state_vector)} does not match input_dim {self.agent_cfg['num_states']}"

        return np.array(flat_state, dtype=np.float32), best_phase_idx
    
    def choose_best_phase(self, state_vector, desra_phase_idx):
        """
        Select the best traffic light phase based on:
        1. DESRA suggestion (if valid and non-zero)
        2. Rule-based comparison: choose the phase that dominates
            others in at least 2 out of 3 features.
        """

        # state_vector is already shaped like [[f1,f2,f3], [f1,f2,f3], ...]
        phases = np.array(state_vector, dtype=float)
        num_phases = len(phases)

        # --- 1. Check DESRA suggestion ---
        if desra_phase_idx is not None and 0 <= desra_phase_idx < num_phases:
            desra_phase = phases[desra_phase_idx]
            if not np.allclose(desra_phase, [0., 0., 0.]):  # reject [0,0,0]
                return desra_phase_idx

        # --- 2. Rule-based selection (2/3 better elements) ---
        best_idx = 0
        for i in range(1, num_phases):
            # Count how many features of phase i are better than current best
            better_count = np.sum(phases[i] > phases[best_idx])
            if better_count >= 2:
                best_idx = i

        return best_idx

    def get_reward(self, tl_id: str, phase: str) -> float:
        """Reward = decrease in LOCAL waiting time and queue length for this specific traffic light.
        Uses the difference in waiting time and queue length before and after the action to provide
        proper credit assignment for multi-agent learning."""

        weight = self.agent_cfg["weight"]
        
        # Get current waiting time and queue length for this specific traffic light
        current_local_wait = 0.0
        current_local_queue = 0.0
        tl_data = None
        for tl in self.traffic_lights:
            if tl["id"] == tl_id:
                tl_data = tl
                break
        
        if tl_data is None:
            return 0.0
            
        current_local_wait = TrafficMetrics.get_mean_waiting_time(tl_data)
        # current_local_queue = TrafficMetrics.get_sum_queue_length(tl_data)
        
        # Initialize per-TL waiting time and queue length tracking if not exists
        if not hasattr(self, 'prev_local_wait'):
            self.prev_local_wait = {}
        # if not hasattr(self, 'prev_local_queue'):
        #     self.prev_local_queue = {}
        
        # First calculation for this TL returns 0 (no baseline)
        # if (tl_id not in self.prev_local_wait or self.prev_local_wait[tl_id] is None):
        #     reward = 0.0
        # else:
        # Reward = reduction in waiting time and queue length (positive if decreased)
        # waiting_reduction = current_local_wait
        # queue_reduction = self.prev_local_queue[tl_id] - current_local_queue

        # waiting_norm = self.waiting_time_normalizer.normalize(current_local_wait)
        # queue_norm = self.queue_length_normalizer.normalize(queue_reduction)
        
        # Scale rewards based on magnitude to improve learning signal
        # Use square root to reduce impact of extreme values
        # waiting_reward = 0.0
        # if waiting_reduction > 0:
        #     waiting_reward = -waiting_reduction * weight["waiting_time"]  # Positive reward for improvement
        # elif waiting_reduction < 0:
        #     waiting_reward = waiting_reduction * weight["waiting_time"]  # Negative reward for worsening
        
        # queue_reward = 0.0
        # if queue_reduction > 0:
        #     queue_reward = -queue_norm * weight["queue_length"]  # Positive reward for queue reduction (weighted less than waiting time)
        # elif queue_reduction < 0:
        #     queue_reward = queue_norm * weight["queue_length"]  # Negative reward for queue increase
        
        # Combined reward: waiting time has higher weight than queue length
        # reward = waiting_reward + queue_reward
        reward = -current_local_wait
        
        # Update snapshots for this traffic light
        self.prev_local_wait[tl_id] = current_local_wait
        # self.prev_local_queue[tl_id] = current_local_queue
        return float(reward)

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
        print("Generating plots at episode", episode, "...")
        for metric, data in avg_history.items():
            # Save data with correct naming convention for visualization  
            self.visualization.save_data(
                data=data,
                filename=f"skrl_dqn_{metric}_avg_episode_{episode}",
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
            "reward", "queue_length", "travel_delay", "waiting_time", "outflow", "junction_throughput", "stopped_vehicles"
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

    def print_desra_usage_summary(self):
        """Print summary of DESRA vs DQN action selection usage"""
        print("\n=== DESRA-DQN Hybrid Usage Summary ===")
        total_desra = sum(self.desra_action_count.values())
        total_dqn = sum(self.dqn_action_count.values())
        total_actions = total_desra + total_dqn
        
        if total_actions > 0:
            desra_percent = (total_desra / total_actions) * 100
            dqn_percent = (total_dqn / total_actions) * 100
            
            print(f"Overall Action Selection:")
            print(f"  DESRA guidance: {total_desra} actions ({desra_percent:.1f}%)")
            print(f"  DQN decisions:  {total_dqn} actions ({dqn_percent:.1f}%)")
            print(f"  Total actions:  {total_actions}")
            
            print(f"\nPer Traffic Light:")
            for tl_id in self.traffic_lights:
                tl_id_str = tl_id["id"]
                desra_count = self.desra_action_count.get(tl_id_str, 0)
                dqn_count = self.dqn_action_count.get(tl_id_str, 0)
                tl_total = desra_count + dqn_count
                if tl_total > 0:
                    tl_desra_percent = (desra_count / tl_total) * 100
                    print(f"  {tl_id_str}: {desra_count}/{tl_total} DESRA ({tl_desra_percent:.1f}%)")
        else:
            print("No actions recorded yet")
        
        print("==========================================\n")

    def reset_desra_tracking(self):
        """Reset DESRA usage tracking for next episode"""
        for tl_id in self.desra_action_count:
            self.desra_action_count[tl_id] = 0
            self.dqn_action_count[tl_id] = 0
