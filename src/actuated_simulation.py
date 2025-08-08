import time
import libsumo as traci
import numpy as np
import pandas as pd
import sys
import os

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from sim_utils.traffic_metrics import TrafficMetrics
from visualization import Visualization
from normalizer import Normalizer
from .sumo import SUMO
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
        min_green_time=20,  # Minimum green time before phase can change
        max_green_time=60,  # Maximum green time before forced phase change
        detection_threshold=2,  # Minimum queue difference to trigger phase change
    ):
        self.max_steps = max_steps
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.visualization = visualization
        self.epoch = epoch
        self.path = path
        self.save_interval = save_interval

        # Actuated control parameters
        self.min_green_time = min_green_time
        self.max_green_time = max_green_time
        self.detection_threshold = detection_threshold
        self.interphase_duration = 3  # Yellow time between phases

        # Initialize VehicleTracker for logging vehicle statistics
        self.vehicle_tracker = VehicleTracker(path=self.path)

        # Add normalizers for reward calculation (same as SKRL simulation)
        self.outflow_rate_normalizer = Normalizer()
        self.queue_length_normalizer = Normalizer()
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
                # step‐accumulators
                "step": {
                    "delay": 0,
                    "time": 0,
                    "density": 0,
                    "outflow": 0,
                    "queue": 0,
                    "waiting": 0,
                    "old_ids": [],
                },
            }

    def get_phase_queue_lengths(self, tl):
        """Get queue lengths for each phase of a traffic light"""
        tl_id = tl["id"]
        phase_queues = {}
        
        for phase_idx, phase_str in enumerate(tl["phase"]):
            total_queue = 0
            movements = self.get_movements_from_phase(tl, phase_str)
            
            for detector_id in movements:
                try:
                    # Get queue length from detector
                    queue_len = traci.lanearea.getLastStepHaltingNumber(detector_id)
                    total_queue += queue_len
                except:
                    # Fallback: get from lane if detector fails
                    try:
                        lane_id = traci.lanearea.getLaneID(detector_id)
                        queue_len = traci.lane.getLastStepHaltingNumber(lane_id)
                        total_queue += queue_len
                    except:
                        pass
            
            phase_queues[phase_idx] = total_queue
            
        return phase_queues

    def select_best_phase(self, tl, current_phase_idx):
        """Select the phase with the longest queue, considering minimum green time"""
        phase_queues = self.get_phase_queue_lengths(tl)
        
        # Find phase with maximum queue
        best_phase_idx = max(phase_queues, key=phase_queues.get)
        best_queue_length = phase_queues[best_phase_idx]
        current_queue_length = phase_queues[current_phase_idx]
        
        # Debug output
        print(f"TL {tl['id']}: Phase queues: {phase_queues}")
        print(f"Current phase {current_phase_idx} queue: {current_queue_length}, Best phase {best_phase_idx} queue: {best_queue_length}")
        
        # Only switch if the difference is significant enough
        if (best_phase_idx != current_phase_idx and 
            best_queue_length > current_queue_length + self.detection_threshold):
            return best_phase_idx, True  # Switch needed
        else:
            return current_phase_idx, False  # Stay in current phase

    def _record_actuated_step_metrics(self, tl):
        """
        Called each sim step for a single traffic light in actuated mode.
        Handles phase switching logic based on queue lengths.
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
                st["phase_start_time"] = self.step
                print(f"TL {tl_id}: Applied new phase {st['current_phase_index']}: {new_phase_str}")
            # Collect metrics even during yellow phase
            self._collect_step_metrics(tl, st)
            return

        # Decrement green time
        if st["green_time_remaining"] > 0:
            st["green_time_remaining"] -= 1

        # Check if we should consider changing phase
        should_check_phase = (
            st["green_time_remaining"] == 0 or  # Minimum green time expired
            (self.step - st["phase_start_time"]) >= self.max_green_time  # Maximum green time reached
        )

        if should_check_phase:
            # Determine best phase based on queue lengths
            new_phase_idx, should_switch = self.select_best_phase(tl, st["current_phase_index"])
            
            # Force switch if maximum green time reached
            if (self.step - st["phase_start_time"]) >= self.max_green_time:
                should_switch = True
                # If no better phase found, cycle to next phase
                if new_phase_idx == st["current_phase_index"]:
                    new_phase_idx = (st["current_phase_index"] + 1) % len(tl["phase"])
                    print(f"TL {tl_id}: Max green time reached, cycling to phase {new_phase_idx}")

            if should_switch and new_phase_idx != st["current_phase_index"]:
                # Start yellow phase transition
                st["current_phase_index"] = new_phase_idx
                st["in_yellow"] = True
                st["yellow_time_remaining"] = self.interphase_duration
                
                # Apply yellow phase (replace G with y)
                current_phase = st["phase"]
                yellow_phase = current_phase.replace("G", "y")
                traci.trafficlight.setRedYellowGreenState(tl_id, yellow_phase)
                print(f"TL {tl_id}: Starting yellow transition to phase {new_phase_idx}")
                
                # Calculate reward for the completed phase
                self._update_metrics_and_reward(tl, st)
            else:
                # Stay in current phase, reset green time
                st["green_time_remaining"] = self.min_green_time

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

        # Update phase tracking
        self.phase[tl_id] = st["phase"]

        # Reset phase metrics
        for key in ["travel_delay_sum", "travel_time_sum", "outflow", "queue_length", "waiting_time"]:
            st[key] = 0

    def _collect_step_metrics(self, tl, st):
        """Collect traffic metrics for current step"""
        tl_id = tl["id"]
        current_phase = st["phase"]

        # Calculate metrics
        new_ids = TrafficMetrics.get_vehicles_in_phase(tl, current_phase)
        outflow = sum(1 for vid in st["old_vehicle_ids"] if vid not in new_ids)
        st["old_vehicle_ids"] = new_ids

        sum_travel_delay = self.get_sum_travel_delay(tl)
        sum_travel_time = self.get_sum_travel_time(tl)
        sum_density = self.get_sum_density(tl)
        sum_queue_length = self.get_sum_queue_length(tl)
        sum_waiting_time = self.get_sum_waiting_time(tl)

        # Accumulate into both TL‐level and step‐level metrics
        st["step"]["outflow"] += outflow
        st["step"]["delay"] += sum_travel_delay
        st["step"]["time"] = sum_travel_time
        st["step"]["density"] += sum_density
        st["step"]["queue"] += sum_queue_length
        st["step"]["waiting"] = sum_waiting_time

        # Update accumulated metrics
        st["outflow"] += outflow
        st["travel_delay_sum"] = sum_travel_delay
        st["travel_time_sum"] = sum_travel_time
        st["queue_length"] = sum_queue_length
        st["waiting_time"] = sum_waiting_time

        # Every 60 steps, flush step‐averages into history
        if self.step % 60 == 0:
            for key, hist in [
                ("delay", "travel_delay"),
                ("time", "travel_time"),
                ("density", "density"),
                ("queue", "queue_length"),
                ("waiting", "waiting_time"),
            ]:
                avg = st["step"][key] / 60.0
                self.history[hist][tl_id].append(avg)
                st["step"][key] = 0

            # Append outflow before resetting
            self.history["outflow"][tl_id].append(st["step"]["outflow"])
            
            # Reset step counters
            st["step"]["outflow"] = 0

    def run(self, episode):
        print("Simulation started (Traffic-Actuated)")
        print("---------------------------------------")
        sim_start = time.time()

        # Initialize per‐TL state for actuated control
        self._init_tl_states(actuated_mode=True)

        num_vehicles = 0
        num_vehicles_out = 0

        # Main simulation loop
        while self.step < self.max_steps:
            # Step simulator
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
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

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
        """Calculate reward for a traffic light action (adapted from SKRL simulation)"""
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

    def reset_history(self):
        """Reset history for next episode"""
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
