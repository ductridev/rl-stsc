"""
Gymnasium-style environment wrapper for SUMO traffic light control.
This module provides environment classes compatible with SKRL and other RL frameworks.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional

try:
    import gymnasium as gym
    from gymnasium import spaces
    Env = gym.Env
    GYMNASIUM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        Env = gym.Env
        GYMNASIUM_AVAILABLE = True
    except ImportError:
        raise ImportError(
            "Neither gymnasium nor gym is available. Please install one of them:\n"
            "For gymnasium: pip install gymnasium\n"
            "For gym: pip install gym"
        )


class TrafficLightEnv(Env):
    """
    Gymnasium environment wrapper for SUMO traffic light control.
    
    This class wraps a single traffic light in the SUMO simulation to provide
    a standard RL environment interface compatible with SKRL and other frameworks.
    """
    
    def __init__(self, simulation, traffic_light_id: str):
        """
        Initialize the traffic light environment.
        
        Args:
            simulation: The SUMO simulation instance
            traffic_light_id: ID of the specific traffic light to control
        """
        super().__init__()
        self.simulation = simulation
        self.traffic_light_id = traffic_light_id
        
        # Action space: number of phases for this traffic light
        num_actions = self.simulation.num_actions[traffic_light_id]
        self.action_space = spaces.Discrete(num_actions)
        
        # Observation space: state vector size
        obs_dim = self.simulation.agent_cfg["num_states"]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.current_state = None
        self.episode_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (initial_observation, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.episode_step = 0
        
        # Get initial state for this traffic light
        tl = next(tl for tl in self.simulation.traffic_lights 
                 if tl["id"] == self.traffic_light_id)
        self.current_state = self.simulation.get_state(tl, tl["phase"][0])
        
        info = {"traffic_light_id": self.traffic_light_id}
        
        return self.current_state, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Note: In this implementation, the actual environment stepping happens
        in the main simulation loop. This method is primarily for reward
        calculation and state updates when called by SKRL.
        
        Args:
            action: Action index to take
            
        Returns:
            Tuple of (next_state, reward, terminated, truncated, info)
        """
        # Get traffic light configuration
        tl = next(tl for tl in self.simulation.traffic_lights 
                 if tl["id"] == self.traffic_light_id)
        
        # Get reward for the action taken
        phase = self.simulation.actions_map[self.traffic_light_id][action]["phase"]
        reward = self.simulation.get_reward(self.traffic_light_id, phase)
        
        # Get next state
        next_state = self.simulation.get_state(tl, phase)
        
        # Check if episode is done
        terminated = self.simulation.step >= self.simulation.max_steps
        truncated = False  # Not using time limits beyond max_steps
        
        self.current_state = next_state
        self.episode_step += 1
        
        info = {
            "traffic_light_id": self.traffic_light_id,
            "episode_step": self.episode_step,
            "action_taken": action,
            "phase": phase
        }
        
        return next_state, reward, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """
        Render the environment (not implemented for SUMO).
        
        Args:
            mode: Rendering mode ('human', 'rgb_array', etc.)
        """
        # SUMO GUI handles visualization, so this is a no-op
        pass
    
    def close(self):
        """Close the environment and clean up resources."""
        # Environment cleanup is handled by the main simulation
        pass
    
    def seed(self, seed: Optional[int] = None) -> List[int]:
        """
        Set random seed for the environment.
        
        Args:
            seed: Random seed
            
        Returns:
            List of seeds used
        """
        if seed is not None:
            np.random.seed(seed)
            return [seed]
        return []


class MultiTrafficLightEnv(Env):
    """
    Multi-agent environment wrapper for multiple traffic lights.
    
    This class can be used for centralized control of multiple traffic lights
    or as a base for multi-agent RL scenarios.
    """
    
    def __init__(self, simulation, traffic_light_ids: List[str]):
        """
        Initialize the multi-traffic light environment.
        
        Args:
            simulation: The SUMO simulation instance
            traffic_light_ids: List of traffic light IDs to control
        """
        super().__init__()
        self.simulation = simulation
        self.traffic_light_ids = traffic_light_ids
        
        # Combined action space: tuple of discrete spaces for each light
        self.action_spaces = {
            tl_id: spaces.Discrete(self.simulation.num_actions[tl_id])
            for tl_id in traffic_light_ids
        }
        
        # Combined observation space
        obs_dim = self.simulation.agent_cfg["num_states"]
        self.observation_spaces = {
            tl_id: spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
            for tl_id in traffic_light_ids
        }
        
        self.current_states = {}
        self.episode_step = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Reset all traffic lights to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)
            
        Returns:
            Tuple of (observations_dict, info_dict)
        """
        if seed is not None:
            np.random.seed(seed)
            
        self.episode_step = 0
        observations = {}
        
        for tl_id in self.traffic_light_ids:
            tl = next(tl for tl in self.simulation.traffic_lights if tl["id"] == tl_id)
            self.current_states[tl_id] = self.simulation.get_state(tl, tl["phase"][0])
            observations[tl_id] = self.current_states[tl_id]
            
        info = {"traffic_light_ids": self.traffic_light_ids, "episode_step": self.episode_step}
        
        return observations, info
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict]:
        """
        Execute one step for all traffic lights.
        
        Args:
            actions: Dictionary mapping traffic light IDs to action indices
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
        """
        observations = {}
        rewards = {}
        terminated = {}
        truncated = {}
        
        for tl_id in self.traffic_light_ids:
            action = actions.get(tl_id, 0)  # Default to action 0 if not provided
            
            # Get traffic light configuration
            tl = next(tl for tl in self.simulation.traffic_lights if tl["id"] == tl_id)
            
            # Get reward and next state
            phase = self.simulation.actions_map[tl_id][action]["phase"]
            rewards[tl_id] = self.simulation.get_reward(tl_id, phase)
            observations[tl_id] = self.simulation.get_state(tl, phase)
            
            # Check termination
            terminated[tl_id] = self.simulation.step >= self.simulation.max_steps
            truncated[tl_id] = False
            
            self.current_states[tl_id] = observations[tl_id]
        
        self.episode_step += 1
        
        info = {
            "episode_step": self.episode_step,
            "actions_taken": actions,
            "global_terminated": all(terminated.values())
        }
        
        return observations, rewards, terminated, truncated, info
    
    def render(self, mode: str = 'human'):
        """Render the environment (not implemented for SUMO)."""
        pass
    
    def close(self):
        """Close the environment and clean up resources."""
        pass


class TrafficLightEnvFactory:
    """
    Factory class for creating traffic light environments.
    """
    
    @staticmethod
    def create_single_env(simulation, traffic_light_id: str) -> TrafficLightEnv:
        """
        Create a single traffic light environment.
        
        Args:
            simulation: SUMO simulation instance
            traffic_light_id: ID of the traffic light
            
        Returns:
            TrafficLightEnv instance
        """
        return TrafficLightEnv(simulation, traffic_light_id)
    
    @staticmethod
    def create_multi_env(simulation, traffic_light_ids: List[str]) -> MultiTrafficLightEnv:
        """
        Create a multi-traffic light environment.
        
        Args:
            simulation: SUMO simulation instance
            traffic_light_ids: List of traffic light IDs
            
        Returns:
            MultiTrafficLightEnv instance
        """
        return MultiTrafficLightEnv(simulation, traffic_light_ids)
    
    @staticmethod
    def create_envs_for_simulation(simulation) -> Dict[str, TrafficLightEnv]:
        """
        Create individual environments for all traffic lights in simulation.
        
        Args:
            simulation: SUMO simulation instance
            
        Returns:
            Dictionary mapping traffic light IDs to environment instances
        """
        envs = {}
        for tl in simulation.traffic_lights:
            tl_id = tl["id"]
            envs[tl_id] = TrafficLightEnv(simulation, tl_id)
        return envs
