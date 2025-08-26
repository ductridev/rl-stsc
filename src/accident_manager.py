# import libsumo as traci
import traci
import random
from shapely.geometry import Point, Polygon

class AccidentManager:
    def __init__(self,  start_step, duration=1000, junction_id_list = [], edge_id_list = [], detection_id_list = []):
        """
        Initialize the AccidentManager class.

        Args:
            junction_id (list): List ID of the junction where the accident may occurs.
            edge_id (list ): List ID of the edge where the accident occurs.
            detection_id (list): List ID of the detectors where the accident may occurs.
            start_step (int): Step at which the accident should start.
            duration (int): Duration for which the accident lasts.
        """
        self.junction_id_list = junction_id_list
        self.edge_id_list = edge_id_list
        self.detection_id_list = detection_id_list
        self.start_step = start_step
        self.duration = duration
        self.accident_active = False
        self.stopped_vehicle = None
        self.accident_created_step = 0

    def count_vehicles_on_junction(self):
        """
        Counts the number of vehicles on a specific junction.

        Returns:
            int: Number of vehicles on the specified junction.
            list: List of vehicle IDs on the junction.
        """
        if not self.junction_id_list:
            print("No junction IDs provided.")
            return 0, []
        vehicle_ids_in_junction = []
        total_vehicles_count = 0
        for junction_id in self.junction_id_list:
            junction_shape = Polygon(traci.junction.getShape(junction_id))
            vehicle_ids = traci.vehicle.getIDList()

            for vehicle_id in vehicle_ids:
                vehicle_position = traci.vehicle.getPosition(vehicle_id)
                vehicle_point = Point(vehicle_position)
                if junction_shape.contains(vehicle_point):
                    total_vehicles_count += 1
                    vehicle_ids_in_junction.append(vehicle_id)

        return total_vehicles_count, vehicle_ids_in_junction
    
    def count_vehicles_on_edge(self):
        """
        Counts the number of vehicles on specific edges.

        Args:
            edge_id (list): ID of the edge to count vehicles on.

        Returns:
            int: Number of vehicles on the specified edge.
            list: List of vehicle IDs on the edge.    
        """
        if not self.edge_id_list:
            print("No edge IDs provided.")
            return 0, []
        vehicle_ids_in_edge_list = []
        total_vehicle_count = 0
        for edge_id in self.edge_id_list:
            vehicle_count = traci.edge.getLastStepVehicleNumber(edge_id)
            total_vehicle_count += vehicle_count
            vehicle_ids_in_edge = traci.edge.getLastStepVehicleIDs(edge_id)
            vehicle_ids_in_edge_list.extend(vehicle_ids_in_edge)
        # print(vehicle_ids_in_edge_list)
        # print(f"Type of vehicle_ids_in_edge_list: {type(vehicle_ids_in_edge_list)}")

        return total_vehicle_count, vehicle_ids_in_edge_list
    
    def count_vehicles_on_detectors(self):
        """
        Counts the number of vehicles on specific detectors.

        Returns:
            int: Number of vehicles on the specified detectors.
            list: List of vehicle IDs on the detectors.    
        """
        if not self.detection_id_list:
            print("No detector IDs provided.")
            return 0, []
        
        vehicle_ids_in_detectors = []
        total_vehicle_count = 0
        
        for detector_id in self.detection_id_list:
            try:
                # Get vehicles in the detector area
                vehicle_count = traci.lanearea.getLastStepVehicleNumber(detector_id)
                total_vehicle_count += vehicle_count
                vehicle_ids = traci.lanearea.getLastStepVehicleIDs(detector_id)
                vehicle_ids_in_detectors.extend(vehicle_ids)
            except traci.TraCIException:
                print(f"Warning: Detector {detector_id} not found or inaccessible.")
                continue

        return total_vehicle_count, vehicle_ids_in_detectors
    
    def random_stop_vehicle(self, vehicle_ids):
        """
        Randomly stops a vehicle in vehicle list.

        Args:
            vehicle_ids (list): List of vehicle IDs to choose from.

        Returns:
            str: ID of the stopped vehicle.
        """
        vehicle_id_stop = random.choice(vehicle_ids)
        if traci.vehicletype.getVehicleClass(traci.vehicle.getTypeID(vehicle_id_stop)) == "pedestrian":
            traci.vehicle.setParameter(vehicle_id_stop, "impatience",  0)
        else:
            traci.vehicle.setSpeed(vehicle_id_stop, 0)
            traci.vehicle.setLaneChangeMode(vehicle_id_stop, 0)

        print(f"🚦 Vehicle {vehicle_id_stop} stopped.")
        return vehicle_id_stop

    def remove_stopped_vehicle(self):
        """
        Removes the stopped vehicle after the accident duration.
        """
        if self.stopped_vehicle is not None:
            try:
                traci.vehicle.remove(self.stopped_vehicle)
                print(f"Vehicle {self.stopped_vehicle} removed after stopping.")
            except traci.TraCIException:
                print(f"Vehicle {self.stopped_vehicle} could not be removed (not found).")

    def is_phase_blocked_by_vehicle(self, tl_id: str, phase_state: str, movements: list[str]) -> bool:
        """
        Check if the given phase string is blocked by the vehicle.
        
        Parameters:
        phase_state (str): Phase string (e.g., "GrGr")
        
        Returns:
        bool: True if blocked, False otherwise
        """
        if self.stopped_vehicle is None:
            return False  # No stopped vehicle to block the phase

        if self.stopped_vehicle not in traci.vehicle.getIDList():
            return False  # Vehicle not present

        road_id = traci.vehicle.getRoadID(self.stopped_vehicle)
        lane_id = traci.vehicle.getLaneID(self.stopped_vehicle)

        # Get controlled links (list of lists of (fromEdge, toEdge, viaLane))
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)

        # Match vehicle’s current road/lane with a controlled link index
        for link_idx, link_entries in enumerate(controlled_links):
            for (from_edge, _, _) in link_entries:
                if lane_id == from_edge:
                    # If this link is GREEN in the given phase → it's blocked
                    if link_idx < len(phase_state) and phase_state[link_idx] in ("G", "g"):
                        return True

        for det in movements:
            if traci.vehicle.getLaneID(self.stopped_vehicle) == traci.lanearea.getLaneID(det):
                return True
        
        # Check collisions
        for col in traci.simulation.getCollisions():
            col_lane = col.lane

            # Match collision lane to a controlled link index
            for link_idx, link_entries in enumerate(controlled_links):
                for (from_edge, _, via_lane) in link_entries:
                    if col_lane in (from_edge, via_lane):
                        # If this link is GREEN in the phase → blocked
                        if link_idx < len(phase_state) and phase_state[link_idx] in ("G", "g"):
                            return True

        return False

    def create_accident(self, current_step):
        """
        Creates an accident at the specified junction, edge, or detector.

        Args:
            current_step (int): Current simulation step.
        """
        if current_step >= self.start_step and current_step < self.start_step + self.duration and not self.accident_active:
            # Collect vehicles from all specified locations
            junction_vehicles = self.count_vehicles_on_junction()[1]
            edge_vehicles = self.count_vehicles_on_edge()[1]
            detector_vehicles = self.count_vehicles_on_detectors()[1]
            
            vehicle_list = junction_vehicles + edge_vehicles + detector_vehicles
            
            if vehicle_list:
                self.stopped_vehicle = self.random_stop_vehicle(vehicle_list)
                self.accident_active = True
                self.accident_created_step = current_step
                print(f"Accident started at step {current_step}.")
                print(f"Vehicle sources: {len(junction_vehicles)} from junctions, "
                      f"{len(edge_vehicles)} from edges, "
                      f"{len(detector_vehicles)} from detectors.")
            else:
                print(f"No vehicles found to stop at step {current_step}.")
        if current_step >= self.accident_created_step + self.duration and current_step <= self.accident_created_step + self.duration * 2 and self.accident_active:
            self.remove_stopped_vehicle()
            self.accident_active = False
            self.stopped_vehicle = None  # Reset stopped vehicle
            print(f"Accident ended at step {current_step}.")