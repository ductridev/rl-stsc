"""
Phase and action management utilities for traffic light control.
"""
from typing import Dict


def index_to_action(index, actions_map):
    """Convert action index to phase string"""
    return actions_map[index]["phase"]


def phase_to_index(phase, actions_map, duration):
    """Convert phase string to action index"""
    for i, action in actions_map.items():
        if action["phase"] == phase:
            return i
    return None
