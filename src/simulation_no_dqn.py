import time
import traci
from src.sumo import SUMO
import numpy as np
from src.accident_manager import AccidentManager

class SimulationBase(SUMO):
    def __init__(self, max_steps, traffic_lights, accident_manager: AccidentManager, epoch=1000):
        self.step = 0
        self.max_steps = max_steps
        self.traffic_lights = traffic_lights
        self.accident_manager = accident_manager
        self.epoch = epoch
        self.history = {
            "travel_speed": {},
            "travel_time": {},
            "density": {},
            "outflow_rate": {},
            "green_time": {},
        }

        self.travel_speed = {}
        self.travel_time = {}
        self.density = {}
        self.outflow_rate = {}

    def run(self):
        print("Simulation started")
        print("Simulating...")
        print("---------------------------------------")

        start = time.time()
        while self.step < self.max_steps:
            traci.simulationStep()
            self.step += 1
            for traffic_light in self.traffic_lights:
                self.travel_speed[traffic_light["id"]] = self.get_avg_speed(traffic_light)
                self.travel_time[traffic_light["id"]] = self.get_avg_travel_time(traffic_light)
                self.density[traffic_light["id"]] = self.get_avg_density(traffic_light)
                self.outflow_rate[traffic_light["id"]] = self.get_avg_outflow(traffic_light)

        traci.close()


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

    def get_avg_outflow(self, traffic_light):
        """
        Calculate the average outflow rate only from detectors on the green phase.
        Assumes traffic_light["green_detectors"] contains the detector IDs for the current green phase.
        """
        outflows = []
        # You may need to adjust this if your data structure is different
        green_detectors = self.get_green_phase_detectors(traffic_light)
        for detector in green_detectors:
            try:
                outflow = traci.lanearea.getLastStepVehicleNumber(detector["id"] if isinstance(detector, dict) else detector)
                outflows.append(outflow)
            except Exception:
                pass
        return np.mean(outflows) if outflows else 0.0

    def get_green_phase_detectors(self, traffic_light):
        """
        Returns a list of detector IDs that are on lanes currently in the green phase
        for the given traffic light.
        """
        tls_id = traffic_light["id"]
        # Get the current phase state (e.g., "GrGr")
        current_phase_state = traci.trafficlight.getRedYellowGreenState(tls_id)
        # Get controlled lanes for this traffic light
        controlled_lanes = traci.trafficlight.getControlledLanes(tls_id)

        green_lanes = [
            lane for lane, signal in zip(controlled_lanes, current_phase_state) if signal.upper() == "G"
        ]

        green_detectors = []
        for detector in traffic_light["detectors"]:
            # detector["lane"] should be the lane id this detector is on
            lane_id = detector.get("lane", None)
            if lane_id in green_lanes:
                green_detectors.append(detector["id"])
        return green_detectors