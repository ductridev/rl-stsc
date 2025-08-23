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
            length = traci.lane.getLastStepLength(lane)
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
            for veh in traci.lane.getLastStepVehicleIDs(lane):
                total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh)
        return total_waiting_time

    @staticmethod
    def get_mean_waiting_time(tl: Dict) -> float:
        """Get mean waiting time per vehicle for a traffic light"""
        total_waiting_time = 0.0
        total_vehicles = 0
        non_stop_vehicles = 0

        for lane in traci.trafficlight.getControlledLanes(tl["id"]):
            lane_waiting_time = 0.0
            for veh in traci.lane.getLastStepVehicleIDs(lane):
                # Get cumulative waiting time for all vehicles in lane
                lane_waiting_time += traci.vehicle.getAccumulatedWaitingTime(veh)
                if traci.vehicle.getAccumulatedWaitingTime(veh) == 0:
                    non_stop_vehicles += 1
                
            # Get number of vehicles currently in lane
            vehicle_count = traci.lane.getLastStepVehicleNumber(lane)

            total_waiting_time += lane_waiting_time
            total_vehicles += vehicle_count

        # Return mean waiting time per vehicle
        return total_waiting_time / (total_vehicles - non_stop_vehicles) if total_vehicles - non_stop_vehicles > 0 else 0.0

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
            total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vid)
        return total_waiting_time

    @staticmethod
    def get_num_vehicles(detector_id: str) -> int:
        """Get number of vehicles in detector"""
        return traci.lanearea.getLastStepVehicleNumber(detector_id)
    
    @staticmethod
    def get_speed(detector_id: str) -> float:
        """Get average speed of vehicles in detector"""
        vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
        total_speed = 0.0
        for vid in vehicle_ids:
            total_speed += traci.vehicle.getSpeed(vid)
        return total_speed
    
    @staticmethod
    def get_occupance(detector_id: str) -> float:
        """Get occupancy of a lane"""
        return traci.lanearea.getLastStepOccupancy(detector_id)

    @staticmethod
    def get_vehicles_in_phase(tl: Dict, phase_str: str) -> List[str]:
        """Get vehicle IDs in a phase"""
        lane_idxs = [detector["id"] for detector in tl["detectors"]]

        vehicle_ids = []
        for lane in lane_idxs:
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

    @staticmethod
    def count_vehicles_in_junction(junction_id: str) -> tuple:
        """
        Count the number of vehicles currently in a specific junction using geometric containment.
        Uses the same logic as AccidentManager.count_vehicles_on_junction().
        
        Args:
            junction_id (str): ID of the junction to count vehicles in
            
        Returns:
            tuple: (vehicle_count, vehicle_ids_list)
                - vehicle_count (int): Number of vehicles in the junction
                - vehicle_ids_list (list): List of vehicle IDs in the junction
        """
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            print("Warning: shapely not available. Cannot count vehicles in junction.")
            return 0, []
        
        try:
            # Get junction shape as polygon
            junction_shape = Polygon(traci.junction.getShape(junction_id))
            vehicle_ids = traci.vehicle.getIDList()
            
            vehicle_ids_in_junction = []
            
            for vehicle_id in vehicle_ids:
                try:
                    vehicle_position = traci.vehicle.getPosition(vehicle_id)
                    vehicle_point = Point(vehicle_position)
                    
                    if junction_shape.contains(vehicle_point):
                        vehicle_ids_in_junction.append(vehicle_id)
                except:
                    # Skip vehicles that can't be queried (may have been removed)
                    continue
            
            return len(vehicle_ids_in_junction), vehicle_ids_in_junction
            
        except Exception as e:
            print(f"Error counting vehicles in junction {junction_id}: {e}")
            return 0, []

    @staticmethod
    def count_vehicles_in_multiple_junctions(junction_id_list: List[str]) -> tuple:
        """
        Count the number of vehicles in multiple junctions.
        Uses the same logic as AccidentManager.count_vehicles_on_junction() for multiple junctions.
        
        Args:
            junction_id_list (List[str]): List of junction IDs to count vehicles in
            
        Returns:
            tuple: (total_vehicle_count, all_vehicle_ids_list)
                - total_vehicle_count (int): Total number of vehicles across all junctions
                - all_vehicle_ids_list (list): List of all vehicle IDs in all junctions
        """
        if not junction_id_list:
            print("No junction IDs provided.")
            return 0, []
        
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            print("Warning: shapely not available. Cannot count vehicles in junctions.")
            return 0, []
        
        vehicle_ids_in_all_junctions = []
        total_vehicles_count = 0
        
        for junction_id in junction_id_list:
            try:
                junction_shape = Polygon(traci.junction.getShape(junction_id))
                vehicle_ids = traci.vehicle.getIDList()

                for vehicle_id in vehicle_ids:
                    try:
                        vehicle_position = traci.vehicle.getPosition(vehicle_id)
                        vehicle_point = Point(vehicle_position)
                        
                        if junction_shape.contains(vehicle_point):
                            # Avoid counting the same vehicle multiple times if it's in multiple junctions
                            if vehicle_id not in vehicle_ids_in_all_junctions:
                                total_vehicles_count += 1
                                vehicle_ids_in_all_junctions.append(vehicle_id)
                    except:
                        # Skip vehicles that can't be queried
                        continue
                        
            except Exception as e:
                print(f"Error processing junction {junction_id}: {e}")
                continue

        return total_vehicles_count, vehicle_ids_in_all_junctions
    
    @staticmethod
    def count_vehicles_entering_junction(junction_id: str, previous_vehicles_in_junction: set = None) -> tuple:
        """
        Count the number of vehicles that ENTERED the junction since the last call.
        This avoids counting stuck vehicles multiple times.
        
        Args:
            junction_id (str): ID of the junction to count vehicles entering
            previous_vehicles_in_junction (set): Set of vehicle IDs that were in junction in previous step
            
        Returns:
            tuple: (new_vehicles_count, new_vehicle_ids, current_vehicles_in_junction)
                - new_vehicles_count (int): Number of vehicles that entered since last call
                - new_vehicle_ids (list): List of vehicle IDs that entered
                - current_vehicles_in_junction (set): Current vehicles in junction (for next call)
        """
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            print("Warning: shapely not available. Cannot count vehicles entering junction.")
            return 0, [], set()
        
        if previous_vehicles_in_junction is None:
            previous_vehicles_in_junction = set()
        
        try:
            # Get junction shape as polygon
            junction_shape = Polygon(traci.junction.getShape(junction_id))
            vehicle_ids = traci.vehicle.getIDList()
            
            current_vehicles_in_junction = set()
            
            for vehicle_id in vehicle_ids:
                try:
                    vehicle_position = traci.vehicle.getPosition(vehicle_id)
                    vehicle_point = Point(vehicle_position)
                    
                    if junction_shape.contains(vehicle_point):
                        current_vehicles_in_junction.add(vehicle_id)
                except:
                    # Skip vehicles that can't be queried (may have been removed)
                    continue
            
            # Find vehicles that entered (present now but not in previous step)
            new_vehicles = current_vehicles_in_junction - previous_vehicles_in_junction
            
            return len(new_vehicles), list(new_vehicles), current_vehicles_in_junction
            
        except Exception as e:
            print(f"Error counting vehicles entering junction {junction_id}: {e}")
            return 0, [], set()

    @staticmethod
    def count_stopped_vehicles_for_traffic_light(tl: Dict) -> int:
        """
        Count the average number of stop events per vehicle for a traffic light.
        A stop event is when a vehicle's speed drops below threshold after being above it.
        
        Args:
            tl (Dict): Traffic light configuration with id
            
        Returns:
            int: Average number of stop events per vehicle (rounded to nearest integer)
        """
        # Access the global vehicle stop tracker
        if not hasattr(TrafficMetrics, '_vehicle_stop_tracker'):
            TrafficMetrics._vehicle_stop_tracker = {}
        
        stop_tracker = TrafficMetrics._vehicle_stop_tracker
        speed_threshold = 0.1  # m/s - below this is considered stopped
        total_stop_events = 0
        vehicle_count = 0
        
        try:
            # Get all lanes controlled by this traffic light
            controlled_lanes = traci.trafficlight.getControlledLanes(tl["id"])
            
            for lane in controlled_lanes:
                try:
                    # Get all vehicle IDs on this lane
                    vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                    
                    for vehicle_id in vehicle_ids:
                        try:
                            vehicle_count += 1
                            current_speed = traci.vehicle.getSpeed(vehicle_id)
                            
                            # Initialize tracking for new vehicles
                            if vehicle_id not in stop_tracker:
                                stop_tracker[vehicle_id] = {
                                    'stop_count': 0,
                                    'was_moving': current_speed >= speed_threshold,
                                    'last_seen_step': traci.simulation.getTime()
                                }
                            
                            vehicle_data = stop_tracker[vehicle_id]
                            current_time = traci.simulation.getTime()
                            
                            # Check for stop event: was moving, now stopped
                            is_currently_stopped = current_speed < speed_threshold
                            was_moving = vehicle_data['was_moving']
                            
                            if was_moving and is_currently_stopped:
                                # Vehicle just stopped - increment stop count
                                vehicle_data['stop_count'] += 1
                                vehicle_data['was_moving'] = False
                            elif not was_moving and not is_currently_stopped:
                                # Vehicle resumed movement
                                vehicle_data['was_moving'] = True
                            
                            # Update last seen time
                            vehicle_data['last_seen_step'] = current_time
                            
                            # Add this vehicle's stop count to total
                            total_stop_events += vehicle_data['stop_count']
                            
                        except:
                            # Skip vehicles that can't be accessed
                            pass
                except:
                    # Skip lanes that can't be accessed
                    pass
        except:
            # Skip if traffic light can't be accessed
            pass
        
        # Clean up old vehicles (not seen for more than 60 seconds)
        current_time = traci.simulation.getTime()
        vehicles_to_remove = []
        for vehicle_id, data in stop_tracker.items():
            if current_time - data['last_seen_step'] > 60:
                vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            del stop_tracker[vehicle_id]
        
        # Return average stop events per vehicle (rounded to nearest integer)
        if vehicle_count > 0:
            return round(total_stop_events / vehicle_count)
        else:
            return 0

    @staticmethod
    def reset_vehicle_stop_tracker():
        """
        Reset the vehicle stop tracking data. Should be called at the start of each episode.
        """
        if hasattr(TrafficMetrics, '_vehicle_stop_tracker'):
            TrafficMetrics._vehicle_stop_tracker.clear()
            print("Vehicle stop tracker reset for new episode")


class JunctionVehicleTracker:
    """Helper class to track vehicles entering junctions over time"""
    
    def __init__(self):
        self.previous_vehicles = {}  # junction_id -> set of vehicle IDs
        self.total_entered = {}     # junction_id -> total count of vehicles entered
        self.entered_this_step = {} # junction_id -> vehicles that entered in current step
    
    def update_junction(self, junction_id: str) -> tuple:
        """
        Update tracking for a specific junction and return vehicles that entered this step.
        
        Args:
            junction_id (str): Junction ID to update
            
        Returns:
            tuple: (new_vehicles_count, new_vehicle_ids)
        """
        # Get previous vehicles for this junction
        previous_vehicles = self.previous_vehicles.get(junction_id, set())
        
        # Count new vehicles entering
        new_count, new_vehicles, current_vehicles = TrafficMetrics.count_vehicles_entering_junction(
            junction_id, previous_vehicles
        )
        
        # Update tracking
        self.previous_vehicles[junction_id] = current_vehicles
        
        if junction_id not in self.total_entered:
            self.total_entered[junction_id] = 0
        self.total_entered[junction_id] += new_count
        
        self.entered_this_step[junction_id] = new_vehicles
        
        return new_count, new_vehicles
    
    def get_total_entered(self, junction_id: str) -> int:
        """Get total number of vehicles that have entered this junction"""
        return self.total_entered.get(junction_id, 0)
    
    def get_current_step_entered(self, junction_id: str) -> list:
        """Get vehicles that entered in the current step"""
        return self.entered_this_step.get(junction_id, [])
    
    def reset_junction(self, junction_id: str):
        """Reset tracking for a specific junction"""
        self.previous_vehicles[junction_id] = set()
        self.total_entered[junction_id] = 0
        self.entered_this_step[junction_id] = []
    
    def reset_all(self):
        """Reset tracking for all junctions"""
        self.previous_vehicles.clear()
        self.total_entered.clear()
        self.entered_this_step.clear()
    
    @staticmethod
    def count_stopped_vehicles() -> tuple:
        """
        Count the total number of stopped vehicles in the simulation at the current step.
        A vehicle is considered stopped if its speed is below a threshold (default 0.1 m/s).
        
        Returns:
            tuple: (stopped_count, stopped_vehicle_ids, total_vehicles)
                - stopped_count (int): Number of vehicles that are stopped
                - stopped_vehicle_ids (list): List of IDs of stopped vehicles
                - total_vehicles (int): Total number of vehicles in simulation
        """
        STOPPED_SPEED_THRESHOLD = 0.1  # m/s - vehicles below this speed are considered stopped
        
        try:
            vehicle_ids = traci.vehicle.getIDList()
            stopped_vehicles = []
            
            for vehicle_id in vehicle_ids:
                try:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    if speed <= STOPPED_SPEED_THRESHOLD:
                        stopped_vehicles.append(vehicle_id)
                except:
                    # Skip vehicles that can't be queried (may have been removed)
                    continue
            
            return len(stopped_vehicles), stopped_vehicles, len(vehicle_ids)
            
        except Exception as e:
            print(f"Error counting stopped vehicles: {e}")
            return 0, [], 0
    
    @staticmethod
    def count_stopped_vehicles_in_junction(junction_id: str) -> tuple:
        """
        Count the number of stopped vehicles specifically within a junction area.
        
        Args:
            junction_id (str): ID of the junction to check for stopped vehicles
            
        Returns:
            tuple: (stopped_count, stopped_vehicle_ids, total_in_junction)
                - stopped_count (int): Number of stopped vehicles in the junction
                - stopped_vehicle_ids (list): List of IDs of stopped vehicles in junction
                - total_in_junction (int): Total number of vehicles in the junction
        """
        STOPPED_SPEED_THRESHOLD = 0.1  # m/s
        
        try:
            from shapely.geometry import Point, Polygon
        except ImportError:
            print("Warning: shapely not available. Cannot count stopped vehicles in junction.")
            return 0, [], 0
        
        try:
            # Get junction shape as polygon
            junction_shape = Polygon(traci.junction.getShape(junction_id))
            vehicle_ids = traci.vehicle.getIDList()
            
            vehicles_in_junction = []
            stopped_vehicles_in_junction = []
            
            for vehicle_id in vehicle_ids:
                try:
                    vehicle_position = traci.vehicle.getPosition(vehicle_id)
                    vehicle_point = Point(vehicle_position)
                    
                    if junction_shape.contains(vehicle_point):
                        vehicles_in_junction.append(vehicle_id)
                        
                        # Check if this vehicle is stopped
                        speed = traci.vehicle.getSpeed(vehicle_id)
                        if speed <= STOPPED_SPEED_THRESHOLD:
                            stopped_vehicles_in_junction.append(vehicle_id)
                except:
                    # Skip vehicles that can't be queried
                    continue
            
            return len(stopped_vehicles_in_junction), stopped_vehicles_in_junction, len(vehicles_in_junction)
            
        except Exception as e:
            print(f"Error counting stopped vehicles in junction {junction_id}: {e}")
            return 0, [], 0
