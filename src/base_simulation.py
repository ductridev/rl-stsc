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
            "outflow_rate": {},
            "green_time": {},
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

        while self.step < self.max_steps:
            for traffic_light in self.traffic_lights:
                traffic_light_id = traffic_light["id"]
                current_phase = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
                green_time = self.green_time[traffic_light_id]
                traci.trafficlight.setPhaseDuration(traffic_light_id, green_time)

                old_vehicle_ids = self.get_vehicles_in_phase(traffic_light, current_phase)
                # Do NOT call self.set_green_phase here; let SUMO handle the logic

                travel_speed = 0
                density = 0

                green_time = min(green_time, self.max_steps - self.step)
                for _ in range(green_time):
                    self.accident_manager.create_accident(current_step=self.step)
                    self.step += 1
                    traci.simulationStep()

                    # Get the updated phase in case SUMO changed it
                    current_phase = traci.trafficlight.getRedYellowGreenState(traffic_light_id)
                    new_vehicle_ids = self.get_vehicles_in_phase(traffic_light, current_phase)
                    outflow = sum(
                        1 for vid in old_vehicle_ids if vid not in new_vehicle_ids
                    )

                    travel_speed += self.get_avg_speed(traffic_light)
                    density += self.get_avg_density(traffic_light)
                    old_vehicle_ids = new_vehicle_ids

                self.outflow_rate[traffic_light_id] = (
                    outflow / green_time if green_time > 0 else 0
                )
                self.travel_speed[traffic_light_id] = (
                    travel_speed / green_time if green_time > 0 else 0
                )
                self.travel_time[traffic_light_id] = (
                    self.get_avg_travel_time(traffic_light) / green_time
                    if green_time > 0
                    else 0
                )
                self.density[traffic_light_id] = (
                    density / green_time if green_time > 0 else 0
                )

                self.history["travel_speed"][traffic_light_id].append(
                    self.travel_speed[traffic_light_id]
                )
                self.history["travel_time"][traffic_light_id].append(
                    self.travel_time[traffic_light_id]
                )
                self.history["density"][traffic_light_id].append(
                    self.density[traffic_light_id]
                )
                self.history["outflow_rate"][traffic_light_id].append(
                    self.outflow_rate[traffic_light_id]
                )
                self.history["green_time"][traffic_light_id].append(
                    self.green_time[traffic_light_id]
                )

        simulation_time = time.time() - start
        traci.close()
        print("Simulation ended")
        print("---------------------------------------")
        self.save_metrics(episode=episode)
        return simulation_time

    def save_metrics(self, episode=None):
        """
        Save and plot metrics similar to save_plot in simulation.py.
        If episode is provided, include it in the filename.
        """

        import json


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
            avg_history[metric] = avg_data[-100:]


        # Save and plot averaged metrics
        if episode % 100 == 0:
            print("Generating plots at episode", episode, "...")
            for metric, data in avg_history.items():
                self.visualization.save_data_and_plot(
                    data=data,
                    filename=f"{metric}_avg{'_episode_' + str(episode) if episode is not None else ''}",
                    xlabel="Step",
                    ylabel=metric.replace("_", " ").title(),
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