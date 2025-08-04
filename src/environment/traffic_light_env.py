"""
Environment wrapper for traffic light control using SKRL.
"""
import numpy as np
from typing import Dict
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from sim_utils.phase_manager import index_to_action


class TrafficLightEnvironment:
    """Environment wrapper for traffic light control using SKRL"""

    def __init__(self, simulation_instance):
        self.sim = simulation_instance
        self.current_tl_id = None
        self.current_tl_config = None

    def set_traffic_light(self, tl_id: str, tl_config: Dict):
        """Set the current traffic light being controlled"""
        self.current_tl_id = tl_id
        self.current_tl_config = tl_config

    def get_observation_space_size(self):
        """Get the size of the observation space"""
        return self.sim.agent_cfg["num_states"]

    def get_action_space_size(self):
        """Get the size of the action space for current traffic light"""
        if self.current_tl_id:
            return self.sim.num_actions[self.current_tl_id]
        return 2  # default

    def get_state(self):
        """Get current state for the traffic light"""
        if self.current_tl_id and self.current_tl_config:
            current_phase = self.sim.tl_states[self.current_tl_id]["phase"]
            return self.sim.get_state(self.current_tl_config, current_phase)
        return np.zeros(self.get_observation_space_size())

    def step(self, action):
        """Execute action and return next state, reward, done"""
        if not self.current_tl_id:
            return np.zeros(self.get_observation_space_size()), 0.0, False

        # Convert action to phase
        phase = index_to_action(action, self.sim.actions_map[self.current_tl_id])

        # Get reward
        reward = self.sim.get_reward(self.current_tl_id, phase)

        # Get next state
        next_state = self.sim.get_state(self.current_tl_config, phase)

        # Check if done
        done = self.sim.step >= self.sim.max_steps

        return next_state, reward, done
