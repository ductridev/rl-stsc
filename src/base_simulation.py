import time
import traci
import numpy as np
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

        self.outflow_rate = {}
        self.travel_speed = {}
        self.travel_time = {}
        self.density = {}
        self.green_time = {}


        self.history = {
            "travel_speed": {},
            "travel_time": {},
            "density": {},
            "outflow": {},
            "green_time": {},
            "queue_length": {},
            "waiting_time": {},        
        }

        self.init_state()

    def init_state(self):
        for traffic_light in self.traffic_lights:
            traffic_light_id = traffic_light["id"]
            self.green_time[traffic_light_id] = 20
            self.outflow_rate[traffic_light_id] = 0
            self.travel_speed[traffic_light_id] = 0
            self.travel_time[traffic_light_id] = 0
            self.density[traffic_light_id] = 0

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
                "density_sum": 0,
                "outflow": 0,
                "old_vehicle_ids": [],
                "phase": None,
                "green_time": 0,
                "last_phase": None,
                "step_travel_speed_sum": 0,
                "step_travel_time_sum": 0,
                "step_density_sum": 0,
                "step_outflow": 0,
                "step_queue_length": 0,
                "step_waiting_time": 0,
                "step_old_vehicle_ids": [],
            }
        while self.step < self.max_steps:
            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                #save plot every 60s
                if self.step % 60 == 0:
                    travel_speed_avg = tl_state["step_travel_speed_sum"] / 60
                    travel_time_avg = tl_state["step_travel_time_sum"] / 60
                    density_avg = tl_state["step_density_sum"] / 60
                    outflow_avg = tl_state["step_outflow"] 
                    queue_length_avg = tl_state["step_queue_length"] / 60
                    waiting_time_avg = tl_state["step_waiting_time"] / 60
                    green_time = tl_state["green_time"]

                    self.history["travel_speed"][tl_id].append(travel_speed_avg)
                    self.history["travel_time"][tl_id].append(travel_time_avg)
                    self.history["density"][tl_id].append(density_avg)
                    self.history["outflow"][tl_id].append(outflow_avg)
                    self.history["green_time"][tl_id].append(green_time)
                    self.history["queue_length"][tl_id].append(queue_length_avg)
                    self.history["waiting_time"][tl_id].append(waiting_time_avg)

                    # Reset step metrics
                    tl_state["step_travel_speed_sum"] = 0
                    tl_state["step_travel_time_sum"] = 0
                    tl_state["step_density_sum"] = 0
                    tl_state["step_outflow"] = 0
                    tl_state["step_queue_length"] = 0
                    tl_state["step_waiting_time"] = 0

                current_phase = traci.trafficlight.getRedYellowGreenState(tl_id)
                # If time to choose a new phase 
                if tl_state["green_time_remaining"] == 0 and current_phase != tl_state["last_phase"] and "y" not in current_phase:
                    green_time = self.green_time[tl_id]
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
                    green_time = self.green_time[tl_id]
                    tl_state.update({
                        "green_time": green_time,
                    })
            self.accident_manager.create_accident(current_step=self.step)
            traci.simulationStep()
            self.step += 1

            for tl in self.traffic_lights:
                tl_id = tl["id"]
                tl_state = tl_states[tl_id]

                if tl_state["green_time_remaining"] > 0:
                    phase = tl_state["phase"]
                    new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                    outflow = sum(
                        1 for vid in tl_state["old_vehicle_ids"] if vid not in new_vehicle_ids
                    )
                    tl_state["outflow"] += outflow
                    tl_state["travel_speed_sum"] += self.get_avg_speed(tl)
                    tl_state["travel_time_sum"] += self.get_avg_travel_time(tl)
                    tl_state["density_sum"] += self.get_avg_density(tl)
                    tl_state["old_vehicle_ids"] = new_vehicle_ids
                    tl_state["green_time_remaining"] -= 1

                    # When phase ends, store metrics
                    if tl_state["green_time_remaining"] == 0:
                        green_time = tl_state["green_time"]

                        self.outflow_rate[tl_id] = tl_state["outflow"] / green_time
                        self.travel_speed[tl_id] = tl_state["travel_speed_sum"] / green_time
                        self.travel_time[tl_id] = tl_state["travel_time_sum"] / green_time
                        self.density[tl_id] = tl_state["density_sum"] / green_time

                # step state metrics
                step_new_vehicle_ids = self.get_vehicles_in_phase(tl, phase)
                step_outflow = sum(
                    1 for vid in tl_state["step_old_vehicle_ids"] if vid not in step_new_vehicle_ids
                )
                tl_state["step_travel_speed_sum"] += self.get_avg_speed(tl)
                tl_state["step_travel_time_sum"] += self.get_avg_travel_time(tl)
                tl_state["step_density_sum"] += self.get_avg_density(tl)
                tl_state["step_outflow"] += step_outflow
                tl_state["step_queue_length"] += self.get_avg_queue_length(tl)
                tl_state["step_waiting_time"] += self.get_avg_waiting_time(tl)
                tl_state["step_old_vehicle_ids"] = step_new_vehicle_ids

        simulation_time = time.time() - start
        traci.close()
        print("Simulation ended")
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
            print("Plots at episode", episode, "generated")
            print("---------------------------------------")

    def set_green_phase(self, tlsId, duration, new_phase):
        traci.trafficlight.setPhaseDuration(tlsId, duration)
        traci.trafficlight.setRedYellowGreenState(tlsId, new_phase)

    def get_avg_speed(self, traffic_light):
        speeds = []
        for detector in traffic_light["detectors"]:
            try:
                speed = traci.lanearea.getLastStepMeanSpeed(detector["id"])
                speeds.append(speed)
            except:
                pass
        return np.mean(speeds) if speeds else 0.0

    def get_avg_travel_time(self, traffic_light):
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(traffic_light["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.mean(travel_times) if travel_times else 0.0

    def get_avg_density(self, traffic_light):
        densities = []
        for detector in traffic_light["detectors"]:
            try:
                densities.append(traci.lanearea.getLastStepOccupancy(detector["id"]))
            except:
                pass
        return np.mean(densities) if densities else 0.0

    def get_avg_queue_length(self, traffic_light):
        """
        Get the average queue length of a lane.
        """
        queue_lengths = []
        for detector in traffic_light["detectors"]:
            try:
                queue_lengths.append(traci.lanearea.getLastStepHaltingNumber(detector["id"]))
            except:
                pass
        return np.mean(queue_lengths) if queue_lengths else 0.0

    def get_avg_waiting_time(self, traffic_light):
        """
        Estimate waiting time by summing waiting times of all vehicles in the lane.
        """
        vehicle_ids = []
        for detector in traffic_light["detectors"]:
            vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(detector["id"]))

        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time
    
    def reset_history(self):
        for key in self.history:
            for tl_id in self.history[key]:
                self.history[key][tl_id] = []