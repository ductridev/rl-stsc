import time
import traci
import numpy as np
import pandas as pd
from src.visualization import Visualization
from src.sumo import SUMO
from src.accident_manager import AccidentManager
from src.vehicle_tracker import VehicleTracker


class SimulationBase(SUMO):
    def __init__(
        self,
        max_steps,
        traffic_lights,
        accident_manager: AccidentManager,
        visualization: Visualization,
        epoch=1000,
        path=None,
    ):
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.visualization = visualization
        self.epoch = epoch
        self.path = path

        # Initialize VehicleTracker for logging vehicle statistics
        self.vehicle_tracker = VehicleTracker(path=self.path)

        self.step = 0

        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_delay = {}
        self.travel_time = {}
        self.waiting_time = {}

        self.history = {
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

    def _record_base_step_metrics(self, tl):
        """
        Called each sim step for a single traffic light in base (SUMO‑controlled) mode.
        Returns the outflow count for this step.
        """
        tl_id = tl["id"]
        st = self.tl_states[tl_id]

        # 1) Detect outflow: vehicles that left since last step
        current_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
        new_ids = self.get_vehicles_in_phase(tl, current_phase)
        outflow = sum(1 for v in st["step"]["old_ids"] if v not in new_ids)
        st["step"]["old_ids"] = new_ids

        sum_travel_delay = self.get_sum_travel_delay(tl)
        sum_travel_time = self.get_sum_travel_time(tl)
        sum_density = self.get_sum_density(tl)
        sum_queue_length = self.get_sum_queue_length(tl)
        sum_waiting_time = self.get_sum_waiting_time(tl)

        # 2) Accumulate into both TL‐level and step‐level metrics
        st["step"]["outflow"] += outflow
        st["step"]["delay"] += sum_travel_delay
        st["step"]["time"] = sum_travel_time
        st["step"]["density"] += sum_density
        st["step"]["queue"] += sum_queue_length
        st["step"]["waiting"] = sum_waiting_time

        st["outflow"] += outflow
        st["travel_delay_sum"] = sum_travel_delay
        st["travel_time_sum"] = sum_travel_time
        st["queue_length"] = sum_queue_length
        st["waiting_time"] = sum_waiting_time

        # 3) Every 60 steps, flush step‐averages into history
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

            self.history["outflow"][tl_id].append(st["step"]["outflow"])
            st["step"]["outflow"] = 0

    def run(self, episode):
        print("Simulation started (Base)")
        print("---------------------------------------")
        sim_start = time.time()

        # 1) Initialize per‐TL state
        self._init_tl_states(base_mode=True)

        num_vehicles = 0
        num_vehicles_out = 0

        # 2) Main loop (one traci.simulationStep per iteration)
        while self.step < self.max_steps:
            # 2a) Step simulator
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
        if episode % 10 == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data(
                    data=data,
                    filename=f"base_{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
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
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

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
        """
        Get the average queue length of a lane.
        """
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
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        total_waiting_time = 0.0
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            total_waiting_time += traci.lane.getWaitingTime(lane)

        return total_waiting_time

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
