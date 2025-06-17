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
        for folder in os.listdir(os.path.join(os.getcwd(), 'simulations')):
            if os.path.isdir(os.path.join(os.getcwd(), 'simulations', folder)):
                if os.path.exists(os.path.join(os.getcwd(), 'simulations', folder, 'osm.sumocfg')):
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
        return os.path.exists(os.path.join(os.getcwd(), 'simulations', intersection, 'osm.sumocfg'))
    
    @staticmethod
    def generate_residential_low_demand_routes(
        intersection,
        enable_bicycle=False,
        enable_pedestrian=False,
        enable_motorcycle=False,
        enable_passenger=False,
        enable_truck=False
    ):
        """
        Generate low-demand routes on highway.residential edges for enabled vehicle types.
        Total target: ~8-10 vehicles per 10 minutes across all selected types.
        """
        print("Generating low-demand routes on residential edges...")

        vehicle_configs = [
            ("bike", "bicycle", "bicycle", enable_bicycle),
            ("ped", "pedestrian", "pedestrian", enable_pedestrian),
            ("motorcycle", "motorcycle", "motorcycle", enable_motorcycle),
            ("veh", "passenger", "passenger", enable_passenger),
            ("truck", "truck", "truck", enable_truck)
        ]

        # Total target = 10 vehicles / 600s → 0.01667 veh/s
        # Divide this across the number of enabled types
        enabled_types = [v for v in vehicle_configs if v[3]]
        if not enabled_types:
            print("No vehicle types enabled for residential low demand. Skipping...")
            return

        per_type_rate = 0.0167 / len(enabled_types)

        for prefix, vclass, vehicle_class, _enabled in enabled_types:
            cmd = (
                f'python "%SUMO_HOME%/tools/randomTrips.py" '
                f'-n osm.net.xml.gz '
                f'--fringe-factor 2 '
                f'--insertion-rate {per_type_rate:.5f} '
                f'-o osm.res_{prefix}.trips.xml '
                f'-r osm.res_{prefix}.rou.xml '
                f'-b 0 -e 3600 '
                f'--trip-attributes "departLane=\\"best\\"" '
                f'--fringe-start-attributes "departSpeed=\\"max\\"" '
                f'--validate --remove-loops '
                f'--via-edge-types highway.residential '
                f'--vehicle-class {vehicle_class} '
                f'--vclass {vclass} '
                f'--prefix res_{prefix} '
                f'--min-distance 150'
            )
            os.system(cmd)
            print(f"Residential {vehicle_class} routes generated")
    
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
        
        original_path = os.getcwd()
        try:
            intersection_path = os.path.join(os.getcwd(), 'simulations', intersection)
            os.chdir(intersection_path)
            Intersection.generate_residential_low_demand_routes(intersection, enable_bicycle, enable_pedestrian, enable_motorcycle, enable_passenger, enable_truck)
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
        finally:
            os.chdir(original_path)

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
        
        os.system('python "%SUMO_HOME%/tools/randomTrips.py" -n osm.net.xml.gz --fringe-factor 2 --insertion-density 60 -o osm.bicycle.trips.xml -r osm.bicycle.rou.xml -b 0 -e 3600 --trip-attributes "departLane=""best""" --fringe-start-attributes "departSpeed=""max""" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class bicycle --vclass bicycle --prefix bike --max-distance 8000')
        print("Bicycle routes generated")
    
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
        
        os.system('python "%SUMO_HOME%/tools/randomTrips.py" -n osm.net.xml.gz --fringe-factor 1 --insertion-density 100 -o osm.pedestrian.trips.xml -r osm.pedestrian.rou.xml -b 0 -e 3600 --vehicle-class pedestrian --prefix ped --pedestrians --max-distance 2000')
        print("Pedestrian routes generated")

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
        
        os.system('python "%SUMO_HOME%/tools/randomTrips.py" -n osm.net.xml.gz --fringe-factor 2 --insertion-density 40 -o osm.motorcycle.trips.xml -r osm.motorcycle.rou.xml -b 0 -e 3600 --trip-attributes "departLane=""best""" --fringe-start-attributes "departSpeed=""max""" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class motorcycle --vclass motorcycle --prefix motorcycle --max-distance 1200')
        print("Motorcycle routes generated")

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
        
        os.system('python "%SUMO_HOME%/tools/randomTrips.py" -n osm.net.xml.gz --fringe-factor 5 --insertion-density 120 -o osm.passenger.trips.xml -r osm.passenger.rou.xml -b 0 -e 3600 --trip-attributes "departLane=""best""" --fringe-start-attributes "departSpeed=""max""" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class passenger --vclass passenger --prefix veh --min-distance 300 --min-distance.fringe 100 --allow-fringe.min-length 10000 --lanes')
        print("Passenger routes generated")

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
        
        os.system('python "%SUMO_HOME%/tools/randomTrips.py" -n osm.net.xml.gz --fringe-factor 5 --insertion-density 80 -o osm.truck.trips.xml -r osm.truck.rou.xml -b 0 -e 3600 --trip-attributes "departLane=""best""" --fringe-start-attributes "departSpeed=""max""" --validate --remove-loops --via-edge-types highway.motorway,highway.motorway_link,highway.trunk_link,highway.primary_link,highway.secondary_link,highway.tertiary_link --vehicle-class truck --vclass truck --prefix truck --min-distance 600 --min-distance.fringe 100 --allow-fringe.min-length 10000 --lanes')
        print("Truck routes generated")