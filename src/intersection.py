import os
import traci

class Intersection:
    @staticmethod
    def get_all_intersections_with_sumo():
        """
        Get all intersections inside intersections folder with sumo.
        This is done by iterating through the data and checking for intersections.
        Args:
            data: input data to check for intersections
        Returns:
            list: list of all intersections found in the data
        """
        # Get all intersections inside intersections folder
        intersections = []
        for folder in os.listdir(os.path.join(os.getcwd(), 'intersection')):
            if os.path.isdir(os.path.join(os.getcwd(), 'intersection', folder)):
                if os.path.exists(os.path.join(os.getcwd(), 'intersection', folder, 'osm.sumocfg')):
                    intersections.append(folder)
        return intersections
    
    @staticmethod
    def check_intersection_exists(intersection):
        """
        Check if an intersection exists in the intersections folder.
        This is done by checking if the folder exists and if it contains a sumocfg file.
        Args:
            intersection: input data to check for intersections
        Returns:
            bool: True if the intersection exists, False otherwise
        """
        return os.path.exists(os.path.join(os.getcwd(), 'intersection', intersection, 'osm.sumocfg'))
    
    @staticmethod
    def generate_routes(intersection, enable_bicycle=False, enable_pedestrian=False, enable_motorcycle=False, enable_passenger=False, enable_truck=False):
        """
        Generate routes for a given intersection.
        This is done by iterating through the data and checking for routes.
        Args:
            intersection: input data to check for routes
            enable_bicycle: boolean to enable bicycle routes
            enable_pedestrian: boolean to enable pedestrian routes
            enable_motorcycle: boolean to enable motorcycle routes
            enable_passenger: boolean to enable passenger routes
            enable_truck: boolean to enable truck routes
        """
        # Check if the intersection exists
        if not Intersection.check_intersection_exists(intersection):
            raise ValueError(f"Intersection {intersection} does not exist.")
        if enable_bicycle:
            Intersection.generate_bicycle_routes(intersection)
        if enable_pedestrian:
            Intersection.generate_pedestrian_routes(intersection)
        if enable_motorcycle:
            Intersection.generate_motorcycle_routes(intersection)
        if enable_passenger:
            Intersection.generate_passenger_routes(intersection)
        if enable_truck:
            Intersection.generate_truck_routes(intersection)

    @staticmethod
    def generate_bicycle_routes(intersection):
        """
        Generate bicycle routes for a given intersection.
        This is done by iterating through the data and checking for bicycle routes.
        Args:
            intersection: input data to check for bicycle routes
        Returns:
            list: list of all bicycle routes found in the data
        """

        os.system('python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 2 --insertion-density 60 -o osm.bicycle.trips.xml -r osm.bicycle.rou.xml -b 0 -e 3600 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class bicycle --vclass bicycle --prefix bike --max-distance 8000')
    
    @staticmethod
    def generate_pedestrian_routes(intersection):
        """
        Generate pedestrian routes for a given intersection.
        This is done by iterating through the data and checking for pedestrian routes.
        Args:
            intersection: input data to check for pedestrian routes
        Returns:
            list: list of all pedestrian routes found in the data
        """
        os.system('python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 1 --insertion-density 100 -o osm.pedestrian.trips.xml -r osm.pedestrian.rou.xml -b 0 -e 3600 --vehicle-class pedestrian --prefix ped --pedestrians --max-distance 2000')

    @staticmethod
    def generate_motorcycle_routes(intersection):
        """
        Generate motorcycle routes for a given intersection.
        This is done by iterating through the data and checking for motorcycle routes.
        Args:
            intersection: input data to check for motorcycle routes
        Returns:
            list: list of all motorcycle routes found in the data
        """
        os.system('python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 2 --insertion-density 40 -o osm.motorcycle.trips.xml -r osm.motorcycle.rou.xml -b 0 -e 3600 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class motorcycle --vclass motorcycle --prefix motorcycle --max-distance 1200')

    @staticmethod
    def generate_passenger_routes(intersection):
        """
        Generate passenger routes for a given intersection.
        This is done by iterating through the data and checking for passenger routes.
        Args:
            intersection: input data to check for passenger routes
        Returns:
            list: list of all passenger routes found in the data
        """
        os.system('python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 5 --insertion-density 120 -o osm.passenger.trips.xml -r osm.passenger.rou.xml -b 0 -e 3600 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --min-distance.fringe 100 --allow-fringe.min-length 10000 --lanes')

    @staticmethod
    def generate_truck_routes(intersection):
        """
        Generate truck routes for a given intersection.
        This is done by iterating through the data and checking for truck routes.
        Args:
            intersection: input data to check for truck routes
        Returns:
            list: list of all truck routes found in the data
        """
        os.system('python "%SUMO_HOME%\tools\randomTrips.py" -n osm.net.xml.gz --fringe-factor 5 --insertion-density 80 -o osm.truck.trips.xml -r osm.truck.rou.xml -b 0 -e 3600 --trip-attributes "departLane=\"best\"" --fringe-start-attributes "departSpeed=\"max\"" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class truck --vclass truck --prefix truck --min-distance 600 --min-distance.fringe 100 --allow-fringe.min-length 10000 --lanes')