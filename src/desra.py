from src.sumo import SUMO

import traci
import numpy as np
from collections import defaultdict, deque
import math


class DESRA(SUMO):
    def __init__(self, interphase_duration=3):
        self.interphase_duration = interphase_duration

        # Rolling buffers for real-time stats
        self.vehicle_counts = defaultdict(lambda: deque())
        self.occupancies = defaultdict(lambda: deque())

        # Estimated real-time parameters
        self.saturation_flow = 0.5
        self.critical_density = 0.6
        self.jam_density = 0.18

    def select_phase_with_desra_hints(
        self,
        traffic_light,
        q_arrivals: dict,
        saturation_flow: float,
        critical_density: float,
        jam_density: float,
    ):
        """
        Select best phase using DESRA hints based on effective outflow.
        """
        best_phase = traffic_light["phase"][0]
        best_green_time = 0
        best_effective_outflow = -1.0

        self.saturation_flow = saturation_flow
        self.critical_density = critical_density
        self.jam_density = jam_density

        print("=== DESRA Hints ===")
        print(f"saturation_flow: {saturation_flow}")
        print(f"critical_density: {critical_density}")
        print(f"jam_density: {jam_density}")

        for phase_str in traffic_light["phase"]:
            movements = self.get_movements_from_phase(traffic_light, phase_str)

            print(f"Phase: {phase_str}")

            # 1) Compute the downstream free space for this phase (xd)
            #    i.e. the most restrictive downstream capacity over all movements.
            xd = float("inf")
            for det in movements:
                lane_len = traci.lanearea.getLength(det)
                free_down = self.get_downstream_queue_length(det)  # in meters
                xd = min(xd, free_down)

            # 2) Find Gsat for each movement, then take the MIN positive → Gmin
            Gmin = float("inf")
            sat_times = {}
            for det in movements:
                x0 = self.get_queue_length(det)  # upstream queue (m)
                q_arr = q_arrivals.get(det, 0.0)  # veh/s
                lane_len = traci.lanearea.getLength(det)  # Δ_j
                Gsat = self.estimate_saturated_green_time(x0, q_arr, xd, lane_len)
                print(
                    f"Det: {det}, Gsat: {Gsat} x0: {x0} q_arr: {q_arr} lane_len: {lane_len} xd: {xd}"
                )
                if Gsat > 0:
                    Gmin = min(Gmin, Gsat)
                sat_times[det] = (Gsat, x0, q_arr, lane_len)

            # if no movement had Gsat>0, skip
            if Gmin == float("inf"):
                continue

            # 3) For this Gmin, compute total effective outflow F_total
            F_total = 0.0
            for det, (Gsat, x0, q_arr, lane_len) in sat_times.items():
                s = self.saturation_flow
                # three‐case formula from Eq. (9):
                if Gsat >= Gmin:
                    # full‐capacity discharge
                    F_total += s * Gmin
                else:
                    # Gsat < Gmin → two subcases
                    # we need to know if the queue‐extent was upstream‐bounded or downstream‐bounded.
                    # but the paper simplifies it to:
                    #  - if xM <= xd then we can use the second branch
                    #  - else the third.
                    # we already computed xM inside estimate_saturated — but for clarity:
                    xM = self.estimate_shockwave_extent(x0, q_arr, lane_len)
                    if xM <= xd:
                        # Eq. (9) middle case: s*Gsat + q_arr*(Gmin - Gsat)
                        F_total += s * Gsat + q_arr * (Gmin - Gsat)
                    else:
                        # Eq. (9) last case: s*Gsat
                        F_total += s * Gsat

            # 4) Compute effective outflow rate v_i
            v_i = F_total / (Gmin + self.interphase_duration)

            print(f"Phase: {phase_str}, Gmin: {Gmin}, F_total: {F_total}, v_i: {v_i}")

            # 5) Track the best
            if v_i > best_effective_outflow:
                best_effective_outflow = v_i
                best_phase = phase_str
                best_green_time = Gmin
            elif v_i == best_effective_outflow and Gmin < best_green_time:
                best_phase = phase_str
                best_green_time = Gmin

        print(
            f"Selected phase {best_phase} with effective outflow {best_effective_outflow} vehicles/s and green time {best_green_time} seconds"
        )

        return best_phase, max(0, int(best_green_time))

    def estimate_shockwave_extent(self, x0, q_arr, link_length):
        """
        Estimate the extent of the shockwave using the shockwave theory.
        Args:
            x0 (float): Upstream queue length **in meters**
            q_arr (float): Arrival flow rate (veh/s)
            link_length (float): Length of upstream link (m)
        Returns:
            float: Shockwave extent (meters)
        """
        s = self.saturation_flow
        kc = self.critical_density
        kj = self.jam_density
        if s <= 0 or kc <= 0 or kj <= 0:
            return 0.0

        denom = kj * (s - q_arr)
        if abs(denom) < 1e-6:
            return link_length

        return x0 * (kj * s - q_arr * kc) / denom

    def estimate_saturated_green_time(self, x0, q_arr, xd, link_length):
        """
        Estimate saturated green time using spillback-aware shockwave theory.
        Args:
            x0 (float): Upstream queue length **in meters**
            q_arr (float): Arrival flow rate (veh/s)
            xd (float): **Min** downstream **free space** in meters
            link_length (float): Length of upstream link (m)
        Returns:
            float: Saturated green time (seconds)
        """
        # — Step 1: unit conversion —
        s = self.saturation_flow
        kc = self.critical_density
        kj = self.jam_density
        if s <= 0 or kc <= 0 or kj <= 0:
            return 0.0

        # — Step 2: x_M per Eq. 3 —
        denom = kj * (s - q_arr)
        if abs(denom) < 1e-6:
            xM = link_length
        else:
            xM = x0 * (kj * s - q_arr * kc) / denom
        xM = max(0.0, min(xM, link_length))

        # — Step 3: the three‐case piecewise green time (Eq. 7):
        #    (remember: Δ ≡ link_length)
        if (xM <= xd) and (xM <= link_length):
            Gsat = xM * kj / s
        elif (link_length <= xM) and (link_length <= xd):
            Gsat = link_length * kj / s
        elif (xd <= xM) and (xd <= link_length):
            Gsat = xd * kj / s
        else:
            Gsat = 0.0

        # print(f"DEBUG: x0={x0}, xd={xd}, link_length={link_length}, xM={xM}, q_arr={q_arr}, xM={xM}, Gsat={Gsat}")

        return Gsat

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

    def get_queue_length(self, detector_id):
        return (
            traci.lanearea.getLastStepOccupancy(detector_id)
            / 100
            * traci.lanearea.getLength(detector_id)
        )

    def get_downstream_queue_length(self, detector_id):
        """
        Returns the available storage (free space in meters) of the downstream link(s)
        for the given area‑detector. Uses the minimum over all downstream lanes.
        """
        lane_id = traci.lanearea.getLaneID(detector_id)
        links = traci.lane.getLinks(lane_id)  # list of (toLane, viaEdge, ...)
        free_spaces = []

        for link in links:
            downstream_lane = link[0]
            # length = traci.lane.getLength(downstream_lane)
            length = 100
            occupancy_pct = traci.lane.getLastStepOccupancy(downstream_lane)
            # occupancy_pct is 0–100%, so queue_length_m = length * (occupancy_pct/100)
            queue_m = length * (occupancy_pct / 100.0)
            free_spaces.append(max(0.0, length - queue_m))

        if free_spaces:
            return min(free_spaces)
        else:
            # no downstream links → assume full storage available
            return traci.lanearea.getLength(detector_id)
