import time
import libsumo as traci
import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sim_utils.traffic_metrics import TrafficMetrics, VehicleCompletionTracker, JunctionVehicleTracker
from visualization import Visualization
from normalizer import Normalizer
from .sumo import SUMO
from accident_manager import AccidentManager
from vehicle_tracker import VehicleTracker


class SimulationBase(SUMO):
    def __init__(
        self,
        agent_cfg,
        max_steps,
        traffic_lights,
        accident_manager: AccidentManager,
        visualization: Visualization,
        epoch=1000,
        path=None,
        save_interval=10,
    ):
        self.max_steps = max_steps
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.visualization = visualization
        self.epoch = epoch
        self.path = path
        self.save_interval = save_interval

        # Initialize VehicleTracker for logging vehicle statistics
        self.vehicle_tracker = VehicleTracker(path=self.path)

        # Initialize completion tracker for episode-end travel time analysis
        self.completion_tracker = VehicleCompletionTracker()

        # Junction vehicle tracking for throughput analysis
        self.junction_tracker = JunctionVehicleTracker()

        # Testing mode flag to disable training operations
        self.testing_mode = False

        # Add normalizers for reward calculation (same as SKRL simulation)
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.num_vehicles_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_delay_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()

        self.step = 0

        # Traffic metrics (same structure as SKRL simulation)
        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_delay = {}
        self.travel_time = {}
        self.waiting_time = {}
        self.phase = {}  # Track current phase for switch penalty

        self.history = {
            "reward": {},
            "travel_delay": {},
            "travel_time": {},
            "density": {},
            "outflow": {},
            "queue_length": {},
            "waiting_time": {},
            "completed_travel_time": {},  # New: track average completed vehicle travel times
            "junction_arrival": {},  # New: track vehicles entering junctions
            "stopped_vehicles": {},  # New: track number of stopped vehicles
        }

        self.init_state()

    def init_state(self):
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            self.queue_length[traffic_light_id] = 0
            self.outflow_rate[traffic_light_id] = 0
            self.travel_delay[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0
            self.phase[traffic_light_id] = traffic_light["phase"][0]  # Initialize with first phase

            for key in self.history:
                self.history[key][traffic_light_id] = []

    def _init_tl_states(self, base_mode=False):
        """
        Prepare self.tl_states for base-mode metrics collection
        """
        self.tl_states = {}
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            self.tl_states[tl_id] = {
                "old_vehicle_ids": [],
                "travel_delay_sum": 0,
                "travel_time_sum": 0,
                "waiting_time": 0,
                "outflow": 0,
                "queue_length": 0,
                "phase": tl["phase"][0],
                "old_phase": tl["phase"][0],
                # step‐accumulators
                "step": {
                    "delay": 0,
                    "time": 0,
                    "density": 0,
                    "outflow": 0,
                    "queue": 0,
                    "waiting": 0,
                    "junction_arrival": 0,
                    "stopped_vehicles": 0,
                    "old_ids": [],
                },
            }

    def _record_base_step_metrics(self, tl):
        """
        Called each sim step for a single traffic light in base (SUMO‑controlled) mode.
        Returns the outflow count for this step.
        """
        tl_id = tl["id"]
        st = self.tl_states[tl_id]

        # 1) Detect outflow: vehicles that left since last step
        current_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
        if current_phase != st["phase"]:
            # Phase change detected, reset old vehicle IDs and update phase tracking
            st["old_vehicle_ids"] = TrafficMetrics.get_vehicles_in_phase(
                tl, current_phase
            )
            # Update phase tracking for reward calculation
            self.phase[tl_id] = current_phase
            st["old_phase"] = st["phase"]  # Store old phase before updating
            st["phase"] = current_phase

            self.queue_length[tl_id] = st["queue_length"]
            self.outflow_rate[tl_id] = st["outflow"]
            self.travel_delay[tl_id] = st["travel_delay_sum"]
            self.travel_time[tl_id] = st["travel_time_sum"]
            self.waiting_time[tl_id] = st["waiting_time"]

            st["outflow"] = 0  # Reset outflow count on phase change

            reward = self.get_reward(tl_id, st["phase"], st["old_phase"])

            self.history["reward"][tl_id].append(reward)

        # Calculate metrics
        new_ids = TrafficMetrics.get_vehicles_in_phase(tl, current_phase)
        
        # Calculate true outflow: vehicles that left the detection zone
        # Outflow = vehicles that were detected before but are not detected now
        old_ids_set = set(st["old_vehicle_ids"])
        new_ids_set = set(new_ids)
        outflow = len(old_ids_set - new_ids_set)  # Vehicles that left
        
        st["old_vehicle_ids"] = new_ids

        sum_travel_delay = TrafficMetrics.get_sum_travel_delay(tl)
        sum_travel_time = TrafficMetrics.get_sum_travel_time(tl)
        sum_density = TrafficMetrics.get_sum_density(tl)
        sum_queue_length = TrafficMetrics.get_sum_queue_length(tl)
        mean_waiting_time = TrafficMetrics.get_mean_waiting_time(tl)
        stopped_vehicles_count = TrafficMetrics.count_stopped_vehicles_for_traffic_light(tl)
        
        # 2) Accumulate into both TL‐level and step‐level metrics
        st["step"]["outflow"] += outflow
        st["step"]["delay"] += sum_travel_delay
        st["step"]["time"] += sum_travel_time  # FIX: accumulate instead of assign
        st["step"]["density"] += sum_density
        st["step"]["queue"] += sum_queue_length
        st["step"]["waiting"] += mean_waiting_time
        st["step"]["stopped_vehicles"] += stopped_vehicles_count

        # Update accumulated metrics
        st["outflow"] += outflow
        st["travel_delay_sum"] = sum_travel_delay
        st["travel_time_sum"] = sum_travel_time
        st["queue_length"] = sum_queue_length
        st["waiting_time"] = mean_waiting_time

        # 3) Every 60 steps, flush step‐averages into history for fairness
        if self.step % 300 == 0:
            
            for key, hist in [
                ("delay", "travel_delay"),
                ("time", "travel_time"),
                ("density", "density"),
                ("queue", "queue_length"),
                ("waiting", "waiting_time"),
                ("stopped_vehicles", "stopped_vehicles"),
            ]:
                avg = st["step"][key] / 300
                self.history[hist][tl_id].append(avg)
                st["step"][key] = 0

            # Append outflow before resetting
            self.history["outflow"][tl_id].append(st["step"]["outflow"])
            
            # Add junction throughput to history (sum over 300 steps)
            if tl_id not in self.history["junction_arrival"]:
                self.history["junction_arrival"][tl_id] = []
            self.history["junction_arrival"][tl_id].append(st["step"]["junction_arrival"])
            
            # Reset step counters
            st["step"]["outflow"] = 0
            st["step"]["junction_arrival"] = 0

    def run(self, episode):
        print("Simulation started (Base)")
        print("---------------------------------------")
        sim_start = time.time()

        # 1) Initialize per‐TL state
        self._init_tl_states(base_mode=True)

        num_vehicles = 0
        num_vehicles_out = 0

        # Warm up 50 steps
        for _ in range(50):
            traci.simulationStep()

        # 2) Main loop (one traci.simulationStep per iteration)
        while self.step < self.max_steps:
            # 2a) Step simulator
            if self.accident_manager:
                self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            num_vehicles_out += traci.simulation.getArrivedNumber()
            self.step += 1

            # Update vehicle tracking statistics
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
                    st["step"]["junction_arrival"] += new_count

            # Print vehicle stats every 1000 steps
            if self.step % 1000 == 0:
                current_stats = self.vehicle_tracker.get_current_stats()
                print(
                    f"Step {self.step}: "
                    f"Running={current_stats['total_running']}, "
                    f"Total Departed={current_stats['total_departed']}, "
                    f"Total Arrived={current_stats['total_arrived']}"
                )

            # 2b) Collect per‐TL metrics and count outflow
            for tl in self.traffic_lights:
                self._record_base_step_metrics(tl)

        # 3) Tear down
        traci.close()
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} through."
        )
        print("---------------------------------------")

        # Print and save vehicle statistics
        self.vehicle_tracker.print_summary("base")
        self.vehicle_tracker.save_logs(episode, "base")
        # Note: vehicle_tracker.reset() moved to train.py after performance tracking

        # Record episode-end completed vehicle travel time for all traffic lights
        episode_avg_travel_time = self.completion_tracker.get_average_total_travel_time()
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            if tl_id not in self.history["completed_travel_time"]:
                self.history["completed_travel_time"][tl_id] = []
            self.history["completed_travel_time"][tl_id].append(episode_avg_travel_time)
        
        print(f"Episode {episode}: {self.completion_tracker.get_completed_count()} vehicles completed, "
              f"average travel time: {episode_avg_travel_time:.2f}s")

        # 4) Save
        self.save_metrics(episode=episode)
        # Note: reset_history() moved to train.py after performance tracking
        self.step = 0

        return time.time() - sim_start

    def save_metrics(self, episode=None):
        """
        Save and plot metrics similar to save_plot in simulation.py.
        If episode is provided, include it in the filename.
        """

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
        if episode % self.save_interval == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                # Save data with correct naming convention for visualization
                self.visualization.save_data(
                    data=data,
                    filename=f"base_{metric}_avg_episode_{episode}",
                )

            # Save metrics DataFrame
            self.save_metrics_to_dataframe(episode=episode)

            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def save_metrics_to_dataframe(self, episode=None):
        """
        Save metrics per traffic light as pandas DataFrame.
        Only saves system metrics: density, outflow, queue_length, travel_delay, travel_time, waiting_time

        Returns:
            pd.DataFrame: DataFrame with columns [traffic_light_id, metric, time_step, value, episode]
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
            "junction_arrival",
            "stopped_vehicles",
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
                                    "simulation_type": "baseline",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}baseline_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}baseline_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"Baseline metrics DataFrame saved to {filename}")

        return df

    def set_green_phase(self, tlsId, duration, new_phase):
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)

    def get_queue_length(self, detector_id):
        """
        Get the queue length of a lane.
        """
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_sum_travel_delay(self, traffic_light) -> float:
        """
        Compute the total travel delay over all approaching lanes for a given traffic light.

        Delay for lane i: D_i = 1 - (average speed / speed limit)
        Total delay: sum of D_i for all lanes.

        Args:
            traffic_light: dict containing traffic light information.

        Returns:
            float: Total delay across all relevant lanes.
        """
        delay_sum = 0.0

        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            speed_limit = traci.lane.getMaxSpeed(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)

            if speed_limit > 0:
                delay = 1.0 - (mean_speed / speed_limit)
                delay_sum += max(0.0, delay)  # avoid negative delay from noisy data

        return delay_sum

    def get_sum_travel_time(self, traffic_light):
        """Calculate average actual travel time of vehicles in the intersection area"""
        total_travel_time = 0.0
        vehicle_count = 0
        
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle_id in vehicle_ids:
                try:
                    # Get actual travel time for each vehicle
                    departure_time = traci.vehicle.getDeparture(vehicle_id)
                    current_time = traci.simulation.getTime()
                    travel_time = current_time - departure_time
                    total_travel_time += travel_time
                    vehicle_count += 1
                except:
                    pass  # Skip vehicles with errors
        
        # Return average travel time per vehicle (or 0 if no vehicles)
        return total_travel_time / vehicle_count if vehicle_count > 0 else 0.0

    def get_sum_density(self, traffic_light):
        """
        Compute density as total vehicles / total lane length (veh/m),
        using true vehicle counts from each lane.
        """
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
        """Get sum of queue lengths for a traffic light"""
        queue_lengths = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            length = traci.lane.getLastStepHaltingNumber(lane)
            if length <= 0:
                queue_lengths.append(0.0)
            else:
                queue_lengths.append(length)
        return np.sum(queue_lengths) if queue_lengths else 0.0

    def get_sum_waiting_time(self, traffic_light):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        total_waiting_time = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            total_waiting_time += traci.lane.getWaitingTime(lane)

        return total_waiting_time

    def get_reward(self, tl_id: str, phase: str, old_phase: str) -> float:
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
        current_local_queue = TrafficMetrics.get_mean_queue_length(tl_data)
        
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

        waiting_norm = self.waiting_time_normalizer.normalize(current_local_wait)
        queue_norm = self.queue_length_normalizer.normalize(current_local_queue)
        outflow_norm = self.outflow_rate_normalizer.normalize(self.outflow_rate[tl_id])
        switch_phase = int(phase != old_phase)
        
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
        reward = -weight["waiting_time"] * waiting_norm - weight["queue_length"] * queue_norm + weight["outflow_rate"] * outflow_norm - weight["switch_phase"] * switch_phase
        # Update snapshots for this traffic light
        self.prev_local_wait[tl_id] = current_local_wait
        # self.prev_local_queue[tl_id] = current_local_queue
        return float(reward)

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
        
        # Reset junction tracking for next episode
        self.junction_tracker.reset_all()
        
        # Reset vehicle stop tracking for next episode
        TrafficMetrics.reset_vehicle_stop_tracker()
