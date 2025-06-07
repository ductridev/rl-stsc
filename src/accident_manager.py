import traci
import random
from shapely.geometry import Point, Polygon

class AccidentManager:
    def __init__(self,  start_step, duration=1000, junction_id_list = [], edge_id_list = [],):
        """
        Initialize the AccidentManager class.

        Args:
            junction_id (list): List ID of the junction where the accident may occurs.
            edge_id (list ): List ID of the edge where the accident occurs.
            start_step (int): Step at which the accident should start.
            duration (int): Duration for which the accident lasts.
        """
        self.junction_id_list = junction_id_list
        self.edge_id_list = edge_id_list
        self.start_step = start_step
        self.duration = duration
        self.accident_active = False
        self.stopped_vehicle = None

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
    
    def random_stop_vehicle(self, vehicle_ids):
        """
        Randomly stops a vehicle in vehicle list.

        Args:
            vehicle_ids (list): List of vehicle IDs to choose from.

        Returns:
            str: ID of the stopped vehicle.
        """
        vehicle_id_stop = random.choice(vehicle_ids)
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
                print(f"✅ Vehicle {self.stopped_vehicle} removed after stopping.")
            except traci.TraCIException:
                print(f"⚠️ Vehicle {self.stopped_vehicle} could not be removed (not found).")

    def create_accident(self, current_step):
        """
        Creates an accident at the specified junction.

        Args:
            current_step (int): Current simulation step.
        """
        if current_step == self.start_step and not self.accident_active:
            vehicle_list = self.count_vehicles_on_junction()[1] + self.count_vehicles_on_edge()[1]
            if vehicle_list:
                self.stopped_vehicle = self.random_stop_vehicle(vehicle_list)
                self.accident_active = True
                print(f"Accident started at step {current_step}.")

        if current_step == self.start_step + self.duration and self.accident_active:
            self.remove_stopped_vehicle()
            self.accident_active = False
            print(f"Accident ended at step {current_step}.")