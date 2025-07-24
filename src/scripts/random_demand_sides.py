import os
import random
import xml.etree.ElementTree as ET


def generate_random_intervals(
    total_duration=3600,
    min_interval=600,
    max_interval=1800,
    base_weight=100.0,
    high_min=200.0,
    high_max=500.0,
    min_active_sides=1,
    max_active_sides=2,
    edge_groups=None,
):
    root = ET.Element("edgedata")
    current_time = 0
    interval_id = 0

    if edge_groups is None:
        raise ValueError(
            "edge_groups must be provided as a dictionary of lists of edges"
        )

    while current_time < total_duration:
        interval_duration = random.randint(min_interval, max_interval)
        begin = current_time
        end = min(current_time + interval_duration, total_duration)

        interval = ET.SubElement(
            root,
            "interval",
            {
                "id": f"interval_{interval_id}",
                "begin": f"{begin:.1f}",
                "end": f"{end:.1f}",
            },
        )

        selected_sides = random.sample(
            list(edge_groups.keys()),
            k=random.randint(min_active_sides, max_active_sides),
        )
        weights = {
            side: (
                random.uniform(high_min, high_max)
                if side in selected_sides
                else base_weight
            )
            for side in edge_groups
        }

        for side, edges in edge_groups.items():
            for edge_id in edges:
                ET.SubElement(
                    interval, "edge", {"id": edge_id, "value": f"{weights[side]:.2f}"}
                )

        current_time = end
        interval_id += 1

    return root


def save_to_same_dir_as_cfg(
    root_element, sumo_cfg_file: str, filename: str = "random_edge_priority"
):
    """
    Save XML near the SUMO config file.

    Args:
        root_element (ET.Element): Root of the XML tree.
        sumo_cfg_file (str): Path to SUMO config file.
        filename (str): Name of the output file.
    """
    directory = os.path.dirname(os.path.abspath(sumo_cfg_file))
    full_path = os.path.join(directory, f"{filename}.src.xml")
    ET.ElementTree(root_element).write(
        full_path, encoding="utf-8", xml_declaration=True
    )
    print(f"Saved to {full_path}")

    return filename
