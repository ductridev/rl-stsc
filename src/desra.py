from src.sumo import SUMO

import traci
import numpy as np

class DESRA(SUMO):
    def __init__(self, interphase_duration=3):
        self.interphase_duration = interphase_duration

        # Default fundamental diagram parameters
        self.saturation_flow = 1800 / 3600  # vehicles per second (e.g., 1800 vph)
        self.jam_density = 0.15  # vehicles per meter
        self.critical_density = 0.03  # vehicles per meter

    def select_phase(self, traffic_light):
        best_phase = None
        best_green_time = 0
        best_effective_outflow = -1

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            green_times = []
            outflows = []

            for detector_id in movements:
                x0 = self.get_queue_length(detector_id)
                q_arr = self.get_arrival_flow(detector_id)
                x0_d = self.get_downstream_queue_length(detector_id)
                link_length = traci.lanearea.getLength(detector_id)

                Gsat = self.estimate_saturated_green_time(
                    x0, q_arr, x0_d, link_length
                )
                if Gsat <= 0:
                    continue

                green_times.append(Gsat)
                outflows.append(self.saturation_flow * Gsat)

            if green_times:
                G_min = min(green_times)
                total_outflow = sum(outflows)
                v_i = total_outflow / (G_min + self.interphase_duration)

                if v_i > best_effective_outflow:
                    best_effective_outflow = v_i
                    best_phase = phase_str
                    best_green_time = G_min

        return best_phase, max(1, int(best_green_time))

    def estimate_saturated_green_time(self, x0, q_arr, x0_d, link_length):
        # Arrival density
        k_arr = q_arr * self.critical_density / self.saturation_flow

        # Max queue extent based on shockwave theory
        numerator = x0 * (self.jam_density * self.saturation_flow - q_arr * self.critical_density)
        denominator = self.jam_density * (self.saturation_flow - q_arr)
        xM = numerator / denominator if denominator > 0 else link_length

        # Nominal green time (shockwave-based)
        if xM <= link_length:
            G_s = xM * self.jam_density / self.saturation_flow
        else:
            G_s = link_length * self.jam_density / self.saturation_flow

        # Available downstream capacity
        x_d = max(0, link_length - x0_d)
        G_d = x_d * self.jam_density / self.saturation_flow

        return min(G_s, G_d)

    def get_movements_from_phase(self, traffic_light, phase_str):
        detectors = [det["id"] for det in traffic_light["detectors"]]
        active_detectors = [
            detectors[i]
            for i, state in enumerate(phase_str)
            if state.upper() == "G" and i < len(detectors)
        ]
        return active_detectors

    def get_queue_length(self, detector_id):
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_downstream_queue_length(self, detector_id):
        return traci.lane.getLastStepHaltingNumber(traci.lane.getLinks(traci.lanearea.getLaneID(detector_id))[0][0])

    def get_arrival_flow(self, detector_id, T=5.0):
        vehicle_count = traci.lanearea.getLastStepVehicleNumber(detector_id)
        return vehicle_count / T  # vehicles per second
