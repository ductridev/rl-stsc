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
from sim_utils.phase_manager import index_to_action, phase_to_index
from accident_manager import AccidentManager
from vehicle_tracker import VehicleTracker


class ActuatedSimulation(SUMO):
    """Traffic-actuated simulation that changes phases based on longest queue length"""
    
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
        min_green_time=10,  # Research standard: 10s minimum green
        max_green_time=40,  # Research standard: 40s maximum green  
        extension_time=5,   # Research standard: 5s extension period
        detection_threshold=1,  # Research standard: 1 vehicle to trigger extension
    ):
        self.max_steps = max_steps
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.visualization = visualization
        self.epoch = epoch
        self.path = path
        self.save_interval = save_interval

        # Research-standard actuated control parameters (following Brilon & Laubert, 1994)
        self.min_green_time = min_green_time      # Research standard: 10s minimum 
        self.max_green_time = max_green_time      # Research standard: 40s maximum
        self.extension_time = extension_time      # Research standard: 5s extension time
        self.detection_threshold = detection_threshold  # Research standard: 1 vehicle
        self.interphase_duration = 3              # Yellow time between phases
        
        # Extension counter tracking per traffic light
        self.extension_counters = {}
        self.vehicles_detected_during_extension = {}

        # Testing mode flag to disable training operations
        self.testing_mode = False

        # Initialize VehicleTracker for logging vehicle statistics
        self.vehicle_tracker = VehicleTracker(path=self.path)
        
        # Initialize completion tracker for episode-end travel time analysis
        self.completion_tracker = VehicleCompletionTracker()

        # Junction vehicle tracking for throughput analysis
        self.junction_tracker = JunctionVehicleTracker()

        # Add normalizers for reward calculation (same as SKRL simulation)
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
        self.travel_time_normalizer = Normalizer()
        self.travel_delay_normalizer = Normalizer()
        self.waiting_time_normalizer = Normalizer()

        self.step = 0
        self.actions_map = {}

        # Traffic metrics (same structure as SKRL simulation)
        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_delay = {}
        self.travel_time = {}
        self.waiting_time = {}
        self.phase = {}  # Track phases

        self.history = {
            "reward": {},
            "travel_delay": {},
            "travel_time": {},
            "density": {},
            "outflow": {},
            "queue_length": {},
            "waiting_time": {},
            "completed_travel_time": {},  # New: track average completed vehicle travel times
            "junction_throughput": {},  # New: track vehicles entering junctions
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
            self.phase[traffic_light_id] = {}
            for phase in traffic_light["phase"]:
                self.phase[traffic_light_id][phase] = 0

            for key in self.history:
                self.history[key][traffic_light_id] = []

            # Initialize the action map
            self.actions_map[traffic_light_id] = {}
            for i, phase in enumerate(traffic_light["phase"]):
                self.actions_map[traffic_light_id][i] = {"phase": phase}

    def _init_tl_states(self, actuated_mode=True):
        """
        Prepare self.tl_states for actuated-mode metrics collection
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
                "current_phase_index": 0,  # Track which phase we're in
                "green_time_remaining": self.min_green_time,
                "yellow_time_remaining": 0,
                "in_yellow": False,
                "phase_start_time": 0,
                "extension_counter": 0,  # Counter for extension mechanism
                "waiting_for_extension": False,  # Whether in extension period
                # step‐accumulators
                "step": {
                    "delay": 0,
                    "time": 0,
                    "density": 0,
                    "outflow": 0,
                    "queue": 0,
                    "waiting": 0,
                    "junction_throughput": 0,
                    "stopped_vehicles": 0,
                    "old_ids": [],
                },
            }
            
            # Initialize extension tracking
            self.extension_counters[tl_id] = 0
            self.vehicles_detected_during_extension[tl_id] = False

    def check_detector_activation(self, tl, current_phase_idx):
        """Check if vehicles are detected on current green phase approaches"""
        tl_id = tl["id"]
        current_phase_str = tl["phase"][current_phase_idx]
        movements = self.get_movements_from_phase(tl, current_phase_str)
        
        vehicles_detected = False
        for detector_id in movements:
            try:
                # Check for vehicle presence on approach lanes
                lane_id = traci.lanearea.getLaneID(detector_id)
                # Check vehicles within ~50m of intersection (approach detection)
                vehicle_count = traci.lane.getLastStepVehicleNumber(lane_id)
                if vehicle_count > 0:
                    vehicles_detected = True
                    break
            except:
                # Fallback: check queue length as proxy for approach vehicles
                try:
                    queue_len = traci.lanearea.getLastStepHaltingNumber(detector_id)
                    if queue_len > 0:
                        vehicles_detected = True
                        break
                except:
                    pass
        
        return vehicles_detected

    def should_extend_green_phase(self, tl, st):
        """Research-standard extension logic: extend if vehicles detected during extension period"""
        tl_id = tl["id"]
        
        # Check if vehicles are detected on current green approaches
        vehicles_detected = self.check_detector_activation(tl, st["current_phase_index"])
        
        # Extension logic following research standard:
        # 1. Wait minimum green time first
        # 2. Then start extension counter 
        # 3. Reset counter if vehicles detected
        # 4. Switch when counter reaches zero OR max green reached
        
        if st["green_time_remaining"] > 0:
            # Still in minimum green period
            return True, "minimum_green"
            
        # Past minimum green - now in extension period
        if not st["waiting_for_extension"]:
            # Start extension period
            st["waiting_for_extension"] = True
            st["extension_counter"] = self.extension_time
            # print(f"TL {tl_id}: Starting extension period ({self.extension_time}s)")
        
        # Check for vehicle detection during extension
        if vehicles_detected:
            # Reset extension counter when vehicles detected
            st["extension_counter"] = self.extension_time
            self.vehicles_detected_during_extension[tl_id] = True
            return True, "extended_due_to_vehicles"
        
        # Countdown extension timer
        if st["extension_counter"] > 0:
            st["extension_counter"] -= 1
            return True, "extension_countdown"
        
        # Extension period expired - time to switch
        return False, "extension_expired"

    def select_next_phase(self, tl, current_phase_idx):
        """Select phase with highest demand (highest queue length + waiting time)"""
        tl_id = tl["id"]
        best_phase_idx = current_phase_idx
        best_demand = -1
        
        # Evaluate demand for each phase
        for phase_idx, phase_str in enumerate(tl["phase"]):
            # Skip current phase to ensure we switch
            if phase_idx == current_phase_idx:
                continue
                
            # Get movements (detectors) for this phase
            movements = self.get_movements_from_phase(tl, phase_str)
            
            # Calculate total demand for this phase
            phase_demand = 0.0
            for detector_id in movements:
                try:
                    # Get queue length (halting vehicles)
                    queue_length = traci.lanearea.getLastStepHaltingNumber(detector_id)
                    
                    # Get waiting time for vehicles in this detector area
                    lane_id = traci.lanearea.getLaneID(detector_id)
                    waiting_time = traci.lane.getWaitingTime(lane_id)
                    
                    # Get number of vehicles for normalization
                    vehicle_count = traci.lanearea.getLastStepVehicleNumber(detector_id)
                    
                    # Combine queue length and normalized waiting time as demand metric
                    # Higher weight on queue length as it represents immediate demand
                    normalized_waiting = waiting_time / max(vehicle_count, 1)  # Avoid division by zero
                    phase_demand += (queue_length * 2.0) + (normalized_waiting / 10.0)  # Scale waiting time
                    
                except Exception:
                    # If detector data unavailable, use lane-based fallback
                    try:
                        for lane in traci.trafficlight.getControlledLanes(tl_id):
                            if str(phase_idx + 1) in lane:  # Simple lane-phase mapping
                                queue_length = traci.lane.getLastStepHaltingNumber(lane)
                                waiting_time = traci.lane.getWaitingTime(lane)
                                vehicle_count = traci.lane.getLastStepVehicleNumber(lane)
                                normalized_waiting = waiting_time / max(vehicle_count, 1)
                                phase_demand += (queue_length * 2.0) + (normalized_waiting / 10.0)
                    except Exception:
                        pass  # Skip this phase if data unavailable
            
            # Update best phase if this has higher demand
            if phase_demand > best_demand:
                best_demand = phase_demand
                best_phase_idx = phase_idx
        
        # Fallback to sequential if no better phase found or all demands are zero
        if best_phase_idx == current_phase_idx or best_demand <= 0:
            best_phase_idx = (current_phase_idx + 1) % len(tl["phase"])
            
        return best_phase_idx

    def _record_actuated_step_metrics(self, tl):
        """
        Research-standard actuated control logic following Brilon & Laubert (1994).
        Implements extension mechanism with vehicle detection.
        """
        tl_id = tl["id"]
        st = self.tl_states[tl_id]

        # Handle yellow phase countdown
        if st["in_yellow"] and st["yellow_time_remaining"] > 0:
            st["yellow_time_remaining"] -= 1
            if st["yellow_time_remaining"] == 0:
                # Yellow phase ended, apply the new phase
                st["in_yellow"] = False
                new_phase_str = tl["phase"][st["current_phase_index"]]
                traci.trafficlight.setRedYellowGreenState(tl_id, new_phase_str)
                st["phase"] = new_phase_str
                st["green_time_remaining"] = self.min_green_time
                st["waiting_for_extension"] = False
                st["extension_counter"] = 0
                st["phase_start_time"] = self.step
                self.vehicles_detected_during_extension[tl_id] = False
                # print(f"TL {tl_id}: Applied new phase {st['current_phase_index']}: {new_phase_str}")
            # Collect metrics even during yellow phase
            self._collect_step_metrics(tl, st)
            return

        # Countdown green time (minimum green period)
        if st["green_time_remaining"] > 0:
            st["green_time_remaining"] -= 1

        should_change_phase = None

        for phase in tl["phase"]:
            if self.phase[tl_id][phase] > 6:
                should_change_phase = phase
                break

        # Check if we should change to the phase never call last 6
        if should_change_phase is None:
            # Check if maximum green time exceeded (force switch)
            phase_duration = self.step - st["phase_start_time"]
            if phase_duration >= self.max_green_time:
                # print(f"TL {tl_id}: Maximum green time ({self.max_green_time}s) reached - forcing phase change")
                should_continue = False
                reason = "max_green_exceeded"
            else:
                # Apply research-standard extension logic
                should_continue, reason = self.should_extend_green_phase(tl, st)

            if not should_continue:
                # Time to switch phases
                new_phase_idx = self.select_next_phase(tl, st["current_phase_index"])
                
                # print(f"TL {tl_id}: Switching from phase {st['current_phase_index']} to {new_phase_idx} (reason: {reason})")
                
                # Calculate reward for the completed phase
                self._update_metrics_and_reward(tl, st)
                
                # Start yellow phase transition
                st["current_phase_index"] = new_phase_idx
                st["in_yellow"] = True
                st["yellow_time_remaining"] = self.interphase_duration
                
                # Apply yellow phase (replace G with y)
                current_phase = st["phase"]
                yellow_phase = current_phase.replace("G", "y")
                traci.trafficlight.setRedYellowGreenState(tl_id, yellow_phase)
                
                # Reset extension tracking
                st["waiting_for_extension"] = False
                st["extension_counter"] = 0
                self.vehicles_detected_during_extension[tl_id] = False

        else:
            # Force switch to the phase that has not been called in the last X steps
            new_phase_idx = phase_to_index(should_change_phase, self.actions_map[tl["id"]], 0)
            
            # print(f"TL {tl_id}: Switching from phase {st['current_phase_index']} to {new_phase_idx} (reason: {reason})")
            
            # Calculate reward for the completed phase
            self._update_metrics_and_reward(tl, st)
            
            # Start yellow phase transition
            st["current_phase_index"] = new_phase_idx
            st["in_yellow"] = True
            st["yellow_time_remaining"] = self.interphase_duration
            
            # Apply yellow phase (replace G with y)
            current_phase = st["phase"]
            yellow_phase = current_phase.replace("G", "y")
            traci.trafficlight.setRedYellowGreenState(tl_id, yellow_phase)
            
            # Reset extension tracking
            st["waiting_for_extension"] = False
            st["extension_counter"] = 0
            self.vehicles_detected_during_extension[tl_id] = False

        # Record metrics for this step
        self._collect_step_metrics(tl, st)

    def _update_metrics_and_reward(self, tl, st):
        """Update metrics and calculate reward when phase ends"""
        tl_id = tl["id"]
        
        # Update class attributes for reward calculation
        self.queue_length[tl_id] = st["queue_length"]
        self.outflow_rate[tl_id] = st["outflow"]
        self.travel_delay[tl_id] = st["travel_delay_sum"]
        self.travel_time[tl_id] = st["travel_time_sum"]
        self.waiting_time[tl_id] = st["waiting_time"]

        # Calculate reward
        reward = self.get_reward(tl_id, st["phase"])
        self.history["reward"][tl_id].append(reward)

        # Reset phase metrics
        for key in ["travel_delay_sum", "travel_time_sum", "outflow", "queue_length", "waiting_time"]:
            st[key] = 0

    def _collect_step_metrics(self, tl, st):
        """Collect traffic metrics for current step"""
        tl_id = tl["id"]
        current_phase = st["phase"]

        for phase in tl["phase"]:
            if phase != current_phase:
                self.phase[tl_id][phase] += 1
            else:
                self.phase[tl_id][phase] = 0

        # Calculate metrics
        new_ids = TrafficMetrics.get_vehicles_in_phase(tl, current_phase)
        outflow = sum(1 for vid in st["old_vehicle_ids"] if vid not in new_ids)
        st["old_vehicle_ids"] = new_ids

        sum_travel_delay = TrafficMetrics.get_sum_travel_delay(tl)
        sum_travel_time = TrafficMetrics.get_sum_travel_time(tl)
        sum_density = TrafficMetrics.get_sum_density(tl)
        sum_queue_length = TrafficMetrics.get_sum_queue_length(tl)
        mean_waiting_time = TrafficMetrics.get_mean_waiting_time(tl)
        stopped_vehicles_count = TrafficMetrics.count_stopped_vehicles_for_traffic_light(tl)
        
        # Accumulate into both TL‐level and step‐level metrics
        st["step"]["outflow"] += outflow
        st["step"]["delay"] += sum_travel_delay
        st["step"]["time"] += sum_travel_time
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

        # Every 300 steps, flush step‐averages into history
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
            if tl_id not in self.history["junction_throughput"]:
                self.history["junction_throughput"][tl_id] = []
            self.history["junction_throughput"][tl_id].append(st["step"]["junction_throughput"])
            
            # Reset step counters
            st["step"]["outflow"] = 0
            st["step"]["junction_throughput"] = 0

    def run(self, episode):
        print("Simulation started (Traffic-Actuated)")
        print("---------------------------------------")
        sim_start = time.time()

        # Initialize per‐TL state for actuated control
        self._init_tl_states(actuated_mode=True)

        num_vehicles = 0
        num_vehicles_out = 0

        # Warm up 50 steps
        for _ in range(50):
            traci.simulationStep()

        # Main simulation loop
        while self.step < self.max_steps:
            # Step simulator
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
                    st["step"]["junction_throughput"] += new_count

            # Print vehicle stats every 1000 steps
            if self.step % 1000 == 0:
                current_stats = self.vehicle_tracker.get_current_stats()
                print(
                    f"Step {self.step}: "
                    f"Running={current_stats['total_running']}, "
                    f"Total Departed={current_stats['total_departed']}, "
                    f"Total Arrived={current_stats['total_arrived']}"
                )

            # Collect per‐TL metrics and handle actuated control
            for tl in self.traffic_lights:
                self._record_actuated_step_metrics(tl)

        # Simulation cleanup
        traci.close()
        print(
            f"Simulation ended — {num_vehicles} departed, {num_vehicles_out} through."
        )
        print("---------------------------------------")

        # Print and save vehicle statistics
        self.vehicle_tracker.print_summary("actuated")
        self.vehicle_tracker.save_logs(episode, "actuated")

        # Record episode-end completed vehicle travel time for all traffic lights
        episode_avg_travel_time = self.completion_tracker.get_average_total_travel_time()
        for tl in self.traffic_lights:
            tl_id = tl["id"]
            if tl_id not in self.history["completed_travel_time"]:
                self.history["completed_travel_time"][tl_id] = []
            self.history["completed_travel_time"][tl_id].append(episode_avg_travel_time)
        
        print(f"Episode {episode}: {self.completion_tracker.get_completed_count()} vehicles completed, "
              f"average travel time: {episode_avg_travel_time:.2f}s")

        # Save metrics
        self.save_metrics(episode=episode)
        self.step = 0

        return time.time() - sim_start

    def save_metrics(self, episode=None):
        """Save and plot metrics similar to save_plot in simulation.py"""
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
                    filename=f"actuated_{metric}_avg_episode_{episode}",
                )

            # Save metrics DataFrame
            self.save_metrics_to_dataframe(episode=episode)

            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def save_metrics_to_dataframe(self, episode=None):
        """Save metrics per traffic light as pandas DataFrame"""
        data_records = []

        # Only collect specified system metrics
        target_metrics = [
            "reward",
            "density",
            "outflow",
            "queue_length",
            "travel_delay",
            "travel_time",
            "waiting_time",
            "junction_throughput",
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
                                    "simulation_type": "actuated",
                                }
                            )

        df = pd.DataFrame(data_records)

        # Save to CSV if path is provided
        if hasattr(self, "path") and self.path:
            filename = (
                f"{self.path}actuated_metrics_episode_{episode}.csv"
                if episode is not None
                else f"{self.path}actuated_metrics.csv"
            )
            df.to_csv(filename, index=False)
            print(f"Actuated metrics DataFrame saved to {filename}")

        return df

    # Traffic metrics calculation methods (same as base simulation)
    def get_sum_travel_delay(self, traffic_light) -> float:
        """Compute the total travel delay over all approaching lanes for a given traffic light"""
        delay_sum = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            speed_limit = traci.lane.getMaxSpeed(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)
            if speed_limit > 0:
                delay = 1.0 - (mean_speed / speed_limit)
                delay_sum += max(0.0, delay)
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
        """Compute density as total vehicles / total lane length (veh/m)"""
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
        return total_veh / total_length if total_length > 0 else 0.0

    def get_movements_from_phase(self, traffic_light, phase_str):
        """Get detector IDs whose street is active (green) in the given phase string"""
        phase_index = traffic_light["phase"].index(phase_str)
        active_street = str(phase_index + 1)
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
            queue_lengths.append(max(0.0, length))
        return np.sum(queue_lengths) if queue_lengths else 0.0

    def get_sum_waiting_time(self, traffic_light):
        """Estimate waiting time by summing waiting times of all vehicles in the lane"""
        total_waiting_time = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            total_waiting_time += traci.lane.getWaitingTime(lane)
        return total_waiting_time

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
        if (tl_id not in self.prev_local_wait or self.prev_local_wait[tl_id] is None):
            reward = 0.0
        else:
            # Reward = reduction in waiting time and queue length (positive if decreased)
            waiting_reduction = self.prev_local_wait[tl_id] - current_local_wait
            # queue_reduction = self.prev_local_queue[tl_id] - current_local_queue

            # waiting_norm = self.waiting_time_normalizer.normalize(waiting_reduction)
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

    def reset_history(self):
        """Reset history for next episode"""
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
        
        # Reset junction tracking for next episode
        self.junction_tracker.reset_all()
        
        # Reset vehicle stop tracking for next episode
        TrafficMetrics.reset_vehicle_stop_tracker()
