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
        for folder in os.listdir(os.path.join(os.getcwd(), "simulations")):
            if os.path.isdir(os.path.join(os.getcwd(), "simulations", folder)):
                if os.path.exists(
                    os.path.join(os.getcwd(), "simulations", folder, "osm.sumocfg")
                ):
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
        return os.path.exists(
            os.path.join(os.getcwd(), "simulations", intersection, "osm.sumocfg")
        )

    @staticmethod
    def generate_residential_demand_routes(
        config,
        simulation_path,
        demand_level="low",  # Options: 'low', 'medium', 'high'
        enable_bicycle=True,
        enable_pedestrian=True,
        enable_motorcycle=True,
        enable_passenger=True,
        enable_truck=True,
    ):
        """
        Generate residential traffic routes based on demand level using pre-defined vehicle counts from config.

        Args:
            config (dict): Configuration dictionary containing 'vehicle_counts'
            simulation_path (str): Path to the simulation folder
            demand_level (str): One of 'low', 'medium', 'high'
            enable_bicycle (bool): Include bicycles
            enable_pedestrian (bool): Include pedestrians
            enable_motorcycle (bool): Include motorcycles
            enable_passenger (bool): Include passenger cars
            enable_truck (bool): Include trucks
        """
        print(f"Generating residential routes for {demand_level} demand...")

        original_path = os.getcwd()

        try:
            intersection_path = os.path.join(
                os.getcwd(), "simulations", simulation_path
            )
            os.chdir(intersection_path)

            if demand_level not in config["vehicle_counts"]:
                print(
                    f"Invalid demand level '{demand_level}'. Must be one of {list(config['vehicle_counts'].keys())}."
                )
                return

            vehicle_counts = config["vehicle_counts"][demand_level]
            simulation_duration = 3600  # 1 hour in seconds

            vehicle_configs = [
                (
                    "motorcycle",
                    "motorcycle",
                    "motorcycle",
                    enable_motorcycle,
                    vehicle_counts.get("motorcycle", 0),
                ),
                (
                    "veh",
                    "passenger",
                    "passenger",
                    enable_passenger,
                    vehicle_counts.get("passenger", 0),
                ),
                (
                    "truck",
                    "truck",
                    "truck",
                    enable_truck,
                    vehicle_counts.get("truck", 0),
                ),
                (
                    "bike",
                    "bicycle",
                    "bicycle",
                    enable_bicycle,
                    vehicle_counts.get("bicycle", 0),
                ),
                (
                    "ped",
                    "pedestrian",
                    "pedestrian",
                    enable_pedestrian,
                    vehicle_counts.get("pedestrian", 0),
                ),
            ]

            enabled_types = [v for v in vehicle_configs if v[3] and v[4] > 0]

            if not enabled_types:
                print(
                    "No vehicle types enabled or no vehicles specified in this demand level."
                )
                return

            total_vehicles = sum(v[4] for v in enabled_types)

            print(
                f"Total vehicles for {demand_level} demand: {total_vehicles} vehicles over {simulation_duration} seconds"
            )

            for prefix, vclass, vehicle_class, _enabled, count in enabled_types:

                # 1. Generate trips file
                trip_cmd = (
                    f'python "%SUMO_HOME%/tools/randomTrips.py" '
                    f"-n osm.net.xml.gz "
                    f"-o osm.res_{prefix}.trips.xml "
                    f"--insertion-rate {count * 2} "
                    f"--begin 0 --end {simulation_duration} "
                    f"--validate --remove-loops "
                    f"--vclass {vclass} "
                    f'--trip-attributes "departLane=\'best\'" '
                    f'--fringe-start-attributes "departSpeed=\'max\'" '
                    f"--prefix res_{prefix} "
                    f"--weights-prefix res_{prefix}_overflow_west"
                )

                if vehicle_class == "pedestrian":
                    trip_cmd += "--via-edge-types footway,path,sidewalk "
                else:
                    f"--vehicle-class {vehicle_class} "

                # 2. Convert trips to routes using duarouter
                route_cmd = (
                    f'duarouter '
                    f'-n osm.net.xml.gz '
                    f'--route-files osm.res_{prefix}.trips.xml '
                    f'-o osm.res_{prefix}.rou.xml'
                )

                # Run both commands
                os.system(trip_cmd)
                os.system(route_cmd)

                print(f"Generated: {count} {vehicle_class} trips")
        finally:
            os.chdir(original_path)
