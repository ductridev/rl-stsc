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
        self.saturation_flow = {}
        self.critical_density = {}
        self.jam_density = {}

    def update_traffic_parameters(self):
        """
        Update real-time traffic parameters for all active detectors.
        Call this every simulation step.
        """
        for detector_id in traci.lanearea.getIDList():
            # Collect new values
            count = traci.lanearea.getLastStepVehicleNumber(detector_id)
            occ = traci.lanearea.getLastStepOccupancy(detector_id)

            # Append to rolling windows
            self.vehicle_counts[detector_id].append(count)
            self.occupancies[detector_id].append(occ)

            # Compute rolling averages
            avg_count = sum(self.vehicle_counts[detector_id]) / len(self.vehicle_counts[detector_id])
            avg_density = sum(self.occupancies[detector_id]) / len(self.occupancies[detector_id]) / 100  # convert to 0-1

            # Update parameters
            self.saturation_flow[detector_id] = min(avg_count, 1)  # veh/s over N seconds
            self.critical_density[detector_id] = min(avg_density, 0.6)
            self.jam_density[detector_id] = max(avg_density, 0.8)  # clipped reasonable max

    def get_current_parameters(self, detector_id):
        return (
            self.saturation_flow.get(detector_id, 1),
            self.critical_density.get(detector_id, 0.6),
            self.jam_density.get(detector_id, 0.8)
        )

    def select_phase_with_desra_hints(self, traffic_light):
        desra_green_times = []
        phase_scores = []
        best_phase = traffic_light["phase"][0]
        best_green_time = 0
        best_effective_outflow = -1

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            green_times = []
            outflows = []
            phase_total_queue = 0
            phase_total_arrival = 0

            for detector_id in movements:
                x0 = self.get_queue_length(detector_id)
                q_arr = self.get_arrival_flow(detector_id)
                x0_d = self.get_downstream_queue_length(detector_id)
                link_length = traci.lanearea.getLength(detector_id)

                phase_total_queue += x0
                phase_total_arrival += q_arr

                if x0 == 0 and q_arr == 0:
                    continue

                # Estimate parameters
                saturation_flow, jam_density, critical_density = self.get_current_parameters(detector_id)

                Gsat = self.estimate_saturated_green_time(
                    x0, q_arr, x0_d, link_length,
                    saturation_flow, jam_density, critical_density
                )

                if Gsat <= 0:
                    continue

                green_times.append(Gsat)
                outflows.append(saturation_flow * Gsat)

            if phase_total_queue + phase_total_arrival < 1:
                desra_green_times.append(0)  # Mark as inactive
                phase_scores.append(0)
                continue

            if green_times:
                G_demand = min(green_times)
                total_outflow = sum(outflows)
                v_i = total_outflow / (G_demand + self.interphase_duration)

                desra_green_times.append(G_demand)
                phase_scores.append(v_i)

                if v_i > best_effective_outflow:
                    best_effective_outflow = v_i
                    best_phase = phase_str
                    best_green_time = G_demand
            else:
                desra_green_times.append(0)
                phase_scores.append(0)

        return best_phase, max(1, int(best_green_time)), desra_green_times

    def estimate_saturated_green_time(self, x0, q_arr, x0_d, link_length, saturation_flow, jam_density, critical_density):
        """
        Estimate saturated green time for a lane using shockwave theory.
        Parameters:
            x0: current queue length (vehicles)
            q_arr: arrival flow (vehicles/s)
            x0_d: downstream queue length (vehicles)
            link_length: length of the lane (meters)
            saturation_flow: estimated saturation flow (vehicles/s)
            jam_density: estimated jam density (veh/m)
            critical_density: estimated critical density (veh/m)
        Returns:
            Estimated green time in seconds
        """

        # Prevent division by zero
        if saturation_flow <= 0 or jam_density <= 0:
            return 0

        # Compute max backward shockwave length (xM)
        numerator = x0 * (jam_density * saturation_flow - q_arr * critical_density)
        denominator = jam_density * (saturation_flow - q_arr)

        # Fallback to full link length if denominator is too small
        if abs(denominator) < 1e-6:
            xM = link_length
        else:
            xM = min(numerator / denominator, link_length)

        # Estimate green time required to discharge queue
        G_s = xM * jam_density / saturation_flow

        # Downstream capacity-based green time
        x_d = max(0, link_length - x0_d)
        G_d = x_d * jam_density / saturation_flow

        # Return the smaller of the two
        return max(0, min(G_s, G_d))

    def get_movements_from_phase(self, traffic_light, phase_str):
        detectors = [det["id"] for det in traffic_light["detectors"]]
        return [detectors[i] for i, state in enumerate(phase_str) if state.upper() == "G" and i < len(detectors)]

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
            link[0] for link in traci.lane.getLinks(lane_id)
            if link[0] != lane_id
        ]

        total = 0
        for upstream_lane in incoming_links:
            total += traci.lane.getLastStepVehicleNumber(upstream_lane)

        return total / len(incoming_links) if incoming_links else 0
