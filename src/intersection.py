import os
import xml.etree.ElementTree as ET

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
        demand_level="low", # Options: 'low', 'medium', 'high'
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
            intersection_path = os.path.join(os.getcwd(), "simulations", simulation_path)
            os.chdir(intersection_path)

            # Automatically detect all demand files per interval
            demand_files = sorted([
                f for f in os.listdir(".")
                if f.startswith("random_edge_priority_interval_") and f.endswith(".src.xml")
            ])

            if not demand_files:
                print("No demand files found matching 'random_edge_priority_interval_*.src.xml'")
                return

            intervals = []
            for demand_file in demand_files:
                tree = ET.parse(demand_file)
                root = tree.getroot()
                for interval in root.findall("interval"):
                    intervals.append((
                        float(interval.get("begin")),
                        float(interval.get("end"))
                    ))

            if demand_level not in config["vehicle_counts"]:
                print(f"Invalid demand level '{demand_level}'. Must be one of {list(config['vehicle_counts'].keys())}.")
                return

            vehicle_counts = config["vehicle_counts"][demand_level]
            simulation_duration = 3600
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
                print("No vehicle types enabled or specified.")
                return

            for prefix, vclass, vehicle_class, _enabled, count in enabled_types:
                print(f"Generating trips for {vehicle_class} (Total: {count})")
                merged_trip_file = f"osm.res_{prefix}.trips.xml"

                with open(merged_trip_file, "w") as merged_file:
                    merged_file.write("<routes>\n")

                    for interval_id, (begin_time, end_time) in enumerate(intervals):
                        demand_file_name = f"random_edge_priority_interval_{interval_id}.src.xml"
                        if not os.path.exists(demand_file_name):
                            print(f"Demand file not found: {demand_file_name}")
                            continue  # Skip if demand file is missing
                        print(f"  Interval {interval_id}: {begin_time}-{end_time}")
                        vehicles_in_interval = int(count / len(intervals))

                        if interval_id == len(intervals) - 1:
                            vehicles_in_interval = count - ((len(intervals) - 1) * int(count / len(intervals)))

                        trip_file = f"osm.res_{prefix}.interval_{interval_id}.trips.xml"
                        trip_cmd = (
                            f'python {original_path}/randomTrips.py '
                            f'-n osm.net.xml.gz '
                            f'-o {trip_file} '
                            f'--begin {begin_time} --end {end_time} '
                            f'--validate --remove-loops '
                            f'--vclass {vclass} '
                            f'--trip-attributes "departLane=\'best\'" '
                            f'--fringe-start-attributes "departSpeed=\'max\'" '
                            f'--prefix res_{prefix}_int{interval_id} '
                            f'--weights-prefix random_edge_priority_interval_{interval_id} '
                        )
                        if interval == 2 or interval_id == 6:
                            trip_cmd += f"--insertion-rate {vehicles_in_interval * 3600 * 4 / (end_time - begin_time)} "
                        else:
                            trip_cmd += f"--insertion-rate {vehicles_in_interval * 3600 / (end_time - begin_time)} "

                        if vehicle_class == "pedestrian":
                            trip_cmd += "--via-edge-types footway,path,sidewalk "
                        else:
                            trip_cmd += f"--vehicle-class {vehicle_class} "


                        # Run randomTrips.py
                        os.system(trip_cmd)

                        # Merge trips into the combined file
                        with open(trip_file, "r") as trip_f:
                            for line in trip_f:
                                if line.strip().startswith("<?xml"):
                                    continue  # Skip XML declaration
                                if "<routes" in line or "</routes" in line:
                                    continue  # Skip <routes> tags
                                merged_file.write(line)

                    merged_file.write("</routes>\n")  # Close routes tag

                # Generate .rou.xml using duarouter for each vehicle class
                route_file = f"osm.res_{prefix}.rou.xml"
                alt_route_file = f"osm.res_{prefix}.rou.alt.xml"
                route_cmd = (
                    f'duarouter '
                    f'-n osm.net.xml.gz '
                    f'--route-files {merged_trip_file} '
                    f'-o {route_file} '
                )
                os.system(route_cmd)

                print(f"Generated route files: {route_file}, {alt_route_file}")

            print("All route files generated.")

        finally:
            os.chdir(original_path)
