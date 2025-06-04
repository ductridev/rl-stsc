import traci
import random
from shapely.geometry import Point, Polygon

class AccidentManager:
    def __init__(self, junction_id, start_step, duration=1000):
        """
        Initialize the AccidentManager class.

        Args:
            junction_id (str): ID of the junction where the accident occurs.
            start_step (int): Step at which the accident should start.
            duration (int): Duration for which the accident lasts.
        """
        self.junction_id = junction_id
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
        junction_shape = Polygon(traci.junction.getShape(self.junction_id))
        vehicles_inside_count = 0
        vehicle_ids_in_junction = []
        vehicle_ids = traci.vehicle.getIDList()

        for vehicle_id in vehicle_ids:
            vehicle_position = traci.vehicle.getPosition(vehicle_id)
            vehicle_point = Point(vehicle_position)
            if junction_shape.contains(vehicle_point):
                vehicles_inside_count += 1
                vehicle_ids_in_junction.append(vehicle_id)

        return vehicles_inside_count, vehicle_ids_in_junction

    def random_stop_vehicle_in_junction(self, vehicle_ids):
        """
        Randomly stops a vehicle in the junction.

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
            vehicle_list = self.count_vehicles_on_junction()[1]
            if vehicle_list:
                self.stopped_vehicle = self.random_stop_vehicle_in_junction(vehicle_list)
                self.accident_active = True
                print(f"Accident started at step {current_step} in junction {self.junction_id}.")

        if current_step == self.start_step + self.duration and self.accident_active:
            self.remove_stopped_vehicle()
            self.accident_active = False
            print(f"Accident ended at step {current_step} in junction {self.junction_id}.")