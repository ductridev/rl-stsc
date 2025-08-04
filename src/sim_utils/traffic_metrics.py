"""
Traffic metrics calculation utilities.
"""

import numpy as np
import traci
from typing import Dict, List


class TrafficMetrics:
    """Utility class for calculating traffic metrics"""

    @staticmethod
    def get_sum_travel_delay(tl: Dict) -> float:
        """Get sum of travel delays for a traffic light"""
        delay_sum = 0.0

        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            speed_limit = traci.lane.getMaxSpeed(lane)
            mean_speed = traci.lane.getLastStepMeanSpeed(lane)

            if speed_limit > 0:
                delay = 1.0 - (mean_speed / speed_limit)
                delay_sum += max(0.0, delay)  # avoid negative delay from noisy data

        return delay_sum

    @staticmethod
    def get_sum_travel_time(tl: Dict) -> float:
        """Get sum of travel times for a traffic light"""
        travel_times = []
        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            travel_times.append(traci.lane.getTraveltime(lane))
        return np.sum(travel_times) if travel_times else 0.0

    @staticmethod
    def get_sum_density(tl: Dict) -> float:
        """Get sum of densities for a traffic light"""
        total_veh = 0
        total_length = 0.0

        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
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

    @staticmethod
    def get_sum_queue_length(tl: Dict) -> float:
        """Get sum of queue lengths for a traffic light"""
        queue_lengths = []
        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            length = traci.lane.getLastStepHaltingNumber(lane)
            if length <= 0:
                queue_lengths.append(0.0)
            else:
                queue_lengths.append(length)
        return np.sum(queue_lengths) if queue_lengths else 0.0

    @staticmethod
    def get_sum_waiting_time(tl: Dict) -> float:
        """Get sum of waiting times for a traffic light"""
        total_waiting_time = 0.0
        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            total_waiting_time += traci.lane.getWaitingTime(lane)
        return total_waiting_time

    @staticmethod
    def get_queue_length(detector_id: str) -> float:
        """Get the queue length of a lane"""
        length = traci.lanearea.getLastStepHaltingNumber(detector_id)
        if length <= 0:
            return 0.0
        return length

    @staticmethod
    def get_waiting_time(detector_id: str) -> float:
        """Estimate waiting time by summing waiting times of all vehicles in the lane"""
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
        total_waiting_time = 0.0
        for vid in vehicle_ids:
            total_waiting_time += traci.vehicle.getWaitingTime(vid)
        return total_waiting_time

    @staticmethod
    def get_num_vehicles(detector_id: str) -> int:
        """Get number of vehicles in detector"""
        return traci.lanearea.getLastStepVehicleNumber(detector_id)

    @staticmethod
    def get_vehicles_in_phase(tl: Dict, phase_str: str) -> List[str]:
        """Get vehicle IDs in a phase"""
        lane_idxs = [detector["id"] for detector in tl["detectors"]]

        green_lanes = [
            lane_idxs[i]
            for i, light_state in enumerate(phase_str)
            if light_state.upper() == "G" and i < len(lane_idxs)
        ]

        vehicle_ids = []
        for lane in green_lanes:
            try:
                vehicle_ids.extend(traci.lanearea.getLastStepVehicleIDs(lane))
            except:
                pass  # Skip any invalid or unavailable lanes

        return vehicle_ids

    @staticmethod
    def get_movements_from_phase(tl: Dict, phase_str: str) -> List[str]:
        """Get movement detectors from a phase"""
        phase_index = tl["phase"].index(phase_str)  # Find index of phase_str
        active_street = str(
            phase_index + 1
        )  # Assuming street "1" is for phase 0, "2" is for phase 1, etc.

        # Collect detector IDs belonging to the active street
        active_detectors = [
            det["id"] for det in tl["detectors"] if det["street"] == active_street
        ]

        return active_detectors
