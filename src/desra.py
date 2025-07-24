from src.sumo import SUMO

import traci
import numpy as np
from collections import defaultdict, deque
import random

class DESRA(SUMO):
    def __init__(self, interphase_duration=3):
        self.interphase_duration = interphase_duration

        # Rolling buffers for real-time stats
        self.vehicle_counts = defaultdict(lambda: deque())
        self.occupancies = defaultdict(lambda: deque())

        # Estimated real-time parameters
        self.saturation_flow = 0.5
        self.critical_density = 0.06
        self.jam_density = 0.18

    def select_phase_with_desra_hints(self, traffic_light):
        """
        Select best phase using DESRA hints based on effective outflow.
        """
        best_phase = traffic_light["phase"][0]
        best_green_time = 0
        best_effective_outflow = -1

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            total_Gsat = 0
            total_outflow = 0
            movement_count = 0

            for detector_id in movements:
                x0 = self.get_queue_length(detector_id)
                q_arr = self.get_arrival_flow(detector_id)
                x0_d = self.get_downstream_queue_length(detector_id)
                link_length = traci.lanearea.getLength(detector_id)

                if x0 == 0 and q_arr == 0:
                    continue

                Gsat = self.estimate_saturated_green_time(x0, q_arr, x0_d, link_length)

                outflow = self.saturation_flow * Gsat
                total_Gsat += Gsat
                total_outflow += outflow
                movement_count += 1

            if movement_count == 0:
                continue

            avg_Gsat = total_Gsat / movement_count
            avg_outflow = total_outflow / movement_count

            # Effective outflow: v_i = s * Gsat / (Gsat + τ)
            v_i = avg_outflow / (avg_Gsat + self.interphase_duration)

            if v_i > best_effective_outflow:
                best_effective_outflow = v_i
                best_phase = phase_str
                best_green_time = avg_Gsat

        return best_phase, max(0, int(best_green_time))

    def estimate_saturated_green_time(self, x0, q_arr, x0_d, link_length):
        """
        Estimate saturated green time using shockwave theory (based on the paper).

        Args:
            x0 (float): Queue length at upstream (vehicles)
            q_arr (float): Arrival flow rate (vehicles/s)
            x0_d (float): Downstream queue length (vehicles)
            link_length (float): Lane length (meters)

        Returns:
            float: Estimated green time (seconds)
        """
        s = self.saturation_flow
        kj = self.jam_density
        kc = self.critical_density

        if s <= 0 or kj <= 0 or kc <= 0:
            return 0

        try:
            w1 = s / kj  # Forward shockwave (discharge)
            w2 = (q_arr - s) / (kc - kj)  # Backward shockwave
        except ZeroDivisionError:
            return 0

        if abs(kc - kj) < 1e-6:
            w2 = 0.01  # prevent division by 0

        denom = w1 + w2
        xM = link_length if abs(denom) < 1e-6 else min(x0 / denom, link_length)

        Gs = xM / w1
        xd = max(0, link_length - x0_d)
        Gd = xd / w1

        green_time = max(0, min(Gs, Gd))  # enforce min green time
        return green_time

    def get_movements_from_phase(self, traffic_light, phase_str):
        """
        Get detector IDs whose street is active (green) in the given phase string.
        """
        phase_index = traffic_light["phase"].index(phase_str)  # Find index of phase_str
        active_street = str(phase_index + 1)  # Assuming street "1" is for phase 0, "2" is for phase 1, etc.

        # Collect detector IDs belonging to the active street
        active_detectors = [
            det["id"]
            for det in traffic_light["detectors"]
            if det["street"] == active_street
        ]

        return active_detectors

    def get_queue_length(self, detector_id):
        return traci.lanearea.getLastStepHaltingNumber(detector_id)

    def get_downstream_queue_length(self, detector_id):
        links = traci.lane.getLinks(traci.lanearea.getLaneID(detector_id))
        if links:
            downstream_lane = links[0][0]
            return traci.lane.getLastStepHaltingNumber(downstream_lane)
        return 0

    def get_arrival_flow(self, detector_id):
        """
        Estimate the arrival flow by analyzing upstream connections.
        This is a best-effort approximation due to SUMO's real-time limits.
        """
        lane_id = traci.lanearea.getLaneID(detector_id)
        incoming_links = [
            link[0] for link in traci.lane.getLinks(lane_id) if link[0] != lane_id
        ]

        total = 0
        for upstream_lane in incoming_links:
            total += traci.lane.getLastStepVehicleNumber(upstream_lane)

        return total / len(incoming_links) if incoming_links else 0
