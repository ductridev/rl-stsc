import time
import traci
import numpy as np
import pandas as pd
from src.visualization import Visualization
from src.sumo import SUMO
from src.accident_manager import AccidentManager


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

        self.step = 0

        self.queue_length = {}
        self.outflow_rate = {}
        self.travel_speed = {}
        self.travel_time = {}
        self.waiting_time = {}

        self.history = {
            "travel_speed": {},
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
            self.travel_speed[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.waiting_time[traffic_light_id] = 0

            for key in self.history:
                self.history[key][traffic_light_id] = []

    def run(self, episode):
        print("Simulation started (Base)")
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
                "old_vehicle_ids": [],
                "phase": None,
                "green_time": 20,
                "last_phase": None,
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

                current_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
                # If time to choose a new phase 
                if tl_state["green_time_remaining"] == 0 and current_phase != tl_state["last_phase"] and "y" not in current_phase:
                    green_time = tl_state["green_time"]
                    green_time = min(green_time, self.max_steps - self.step)
                    traci.trafficlight.setPhaseDuration(tl_id, green_time)
                    # print(f"Setting green time for phase {current_phase} to {green_time} seconds")
                    tl_state.update({
                        "green_time": green_time,
                        "green_time_remaining": green_time,
                        "travel_speed_sum": 0,
                        "travel_time_sum": 0,
                        "density_sum": 0,
                        "outflow": 0,
                        "old_vehicle_ids": self.get_vehicles_in_phase(tl, current_phase),
                        "phase": current_phase,
                        "last_phase": current_phase,
                    })
                elif tl_state["green_time_remaining"] == 0 and current_phase == tl_state["last_phase"]:
                    green_time = tl_state["green_time"]
                    tl_state.update({
                        "green_time": green_time,
                    })
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            num_vehicles += traci.simulation.getDepartedNumber()
            self.step += 1

            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                if self.step % 60 == 0:
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
                        1 for vid in tl_state["old_vehicle_ids"] if vid not in new_vehicle_ids
                    )
                    num_vehicles_out += outflow
                    tl_state["outflow"] += outflow
                    tl_state["travel_speed_sum"] += self.get_sum_speed(tl)
                    tl_state["travel_time_sum"] += self.get_sum_travel_time(tl)
                    tl_state["waiting_time"] += self.get_sum_waiting_time(tl)
                    tl_state["old_vehicle_ids"] = new_vehicle_ids
                    tl_state["queue_length"] += self.get_sum_queue_length(tl)
                    tl_state["green_time_remaining"] -= 1

                    # When phase ends, store metrics
                    if tl_state["green_time_remaining"] == 0:
                        green_time = tl_state["green_time"]

                        self.queue_length[tl_id] = tl_state["queue_length"]
                        self.outflow_rate[tl_id] = tl_state["outflow"] / green_time
                        self.travel_speed[tl_id] = tl_state["travel_speed_sum"]
                        self.travel_time[tl_id] = tl_state["travel_time_sum"] / green_time
                        self.waiting_time[tl_id] = tl_state["waiting_time"]

                # step state metrics
                step_new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                step_outflow = sum(
                    1 for vid in tl_state["step_old_vehicle_ids"] if vid not in step_new_vehicle_ids
                )
                tl_state["step_travel_speed_sum"] += self.get_sum_speed(tl)
                tl_state["step_travel_time_sum"] += self.get_sum_travel_time(tl)
                tl_state["step_density_sum"] += self.get_sum_density(tl)
                tl_state["step_outflow"] += step_outflow
                tl_state["step_queue_length"] += self.get_sum_queue_length(tl)
                tl_state["step_waiting_time"] += self.get_sum_waiting_time(tl)
                tl_state["step_old_vehicle_ids"] = step_new_vehicle_ids

        simulation_time = time.time() - start
        traci.close()
        print("Simulation ended")
        print("Number of vehicles:", num_vehicles)
        print("Number of vehicles get through all intersections:", num_vehicles_out)
        print("---------------------------------------")
        self.save_metrics(episode=episode)
        self.reset_history()
        self.step = 0
        return simulation_time

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

    def get_sum_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

    def get_sum_density(self, traffic_light):
        densities = []
        for detector in traffic_light["detectors"]:
            try:
                densities.append(traci.lanearea.getLastStepOccupancy(detector["id"]))
            except:
                pass
        return np.sum(densities) / 100 if densities else 0.0

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

    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []
