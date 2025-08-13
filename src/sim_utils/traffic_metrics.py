"""
Traffic metrics calculation utilities.
"""

import numpy as np
import libsumo as traci
from typing import Dict, List


class VehicleCompletionTracker:
    """Tracks completed vehicles and their total travel times for episode-end analysis"""
    
    def __init__(self):
        self.completed_vehicles = []
        self.total_travel_times = []
        self.vehicle_departure_times = {}  # Store departure times of active vehicles
        
    def update_completed_vehicles(self):
        """Call each simulation step to track newly arrived vehicles"""
        # First, track departure times of all current vehicles
        current_vehicles = traci.simulation.getDepartedIDList()
        for vehicle_id in current_vehicles:
            if vehicle_id not in self.vehicle_departure_times:
                try:
                    departure_time = traci.vehicle.getDeparture(vehicle_id)
                    self.vehicle_departure_times[vehicle_id] = departure_time
                except:
                    # Fallback to current simulation time if getDeparture fails
                    self.vehicle_departure_times[vehicle_id] = traci.simulation.getTime()
        
        # Track vehicles that have arrived at their destination
        arrived_vehicles = traci.simulation.getArrivedIDList()
        current_time = traci.simulation.getTime()
        for vehicle_id in arrived_vehicles:
            if vehicle_id in self.vehicle_departure_times:
                departure_time = self.vehicle_departure_times[vehicle_id]
                travel_time = current_time - departure_time
                
                self.completed_vehicles.append(vehicle_id)
                self.total_travel_times.append(travel_time)
                
                # Remove from tracking dictionary to save memory
                del self.vehicle_departure_times[vehicle_id]
            else:
                print(f"DEBUG: Vehicle {vehicle_id} arrived but departure time not tracked")
    
    def get_average_total_travel_time(self) -> float:
        """Get average total travel time of all completed vehicles"""
        if not self.total_travel_times:
            return 0.0
        return sum(self.total_travel_times) / len(self.total_travel_times)
    
    def get_completed_count(self) -> int:
        """Get number of vehicles that completed their journey"""
        return len(self.completed_vehicles)
    
    def reset(self):
        """Reset for next episode"""
        print(f"DEBUG: Resetting completion tracker. Previous episode had {len(self.completed_vehicles)} completed vehicles.")
        self.completed_vehicles = []
        self.total_travel_times = []
        self.vehicle_departure_times = {}


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
        """Calculate average actual travel time of vehicles in the intersection area"""
        total_travel_time = 0.0
        vehicle_count = 0
        
        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for vehicle_id in vehicle_ids:
                try:
                    # Get actual travel time for each vehicle
                    departure_time = traci.vehicle.getDeparture(vehicle_id)
                    current_time = traci.simulation.getTime()
                    travel_time = current_time - departure_time
                    total_travel_time += travel_time
                    vehicle_count += 1
                except:
                    pass  # Skip vehicles with errors
        
        # Return average travel time per vehicle (or 0 if no vehicles)
        return total_travel_time / vehicle_count if vehicle_count > 0 else 0.0

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
    def get_mean_waiting_time(tl: Dict) -> float:
        """Get mean waiting time per vehicle for a traffic light"""
        total_waiting_time = 0.0
        total_vehicles = 0

        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            # Get cumulative waiting time for all vehicles in lane
            lane_waiting_time = traci.lane.getWaitingTime(lane)
            # Get number of vehicles currently in lane
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)

            total_waiting_time += lane_waiting_time
            total_vehicles += vehicle_count

        # Return mean waiting time per vehicle
        return total_waiting_time / total_vehicles if total_vehicles > 0 else 0.0

    @staticmethod
    def get_completed_vehicles_travel_time() -> float:
        """Get average travel time of vehicles that arrived in the current simulation step"""
        arrived_vehicles = traci.simulation.getArrivedIDList()
        if not arrived_vehicles:
            return 0.0
        
        total_travel_time = 0.0
        valid_vehicles = 0
        
        for vehicle_id in arrived_vehicles:
            try:
                departure_time = traci.vehicle.getDeparture(vehicle_id)
                arrival_time = traci.vehicle.getArrival(vehicle_id)
                travel_time = arrival_time - departure_time
                total_travel_time += travel_time
                valid_vehicles += 1
            except:
                pass  # Skip vehicles with errors
        
        return total_travel_time / valid_vehicles if valid_vehicles > 0 else 0.0

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
