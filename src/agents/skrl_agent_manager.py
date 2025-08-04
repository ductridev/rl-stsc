"""
SKRL agent setup and management for traffic light control.
"""
import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN, DQN_DEFAULT_CONFIG
from skrl.utils import set_seed
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.q_network import QNetwork
from environment.traffic_light_env import TrafficLightEnvironment


class SKRLAgentManager:
    """Manager for SKRL agents and components"""

    def __init__(self, simulation_instance, agent_cfg: Dict, traffic_lights: List[Dict], 
                 updating_target_network_steps: int = 100, device: torch.device = None):
        self.sim = simulation_instance
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.updating_target_network_steps = updating_target_network_steps
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set random seed for reproducibility
        set_seed(42)
        
        # Initialize containers
        self.agents = {}
        self.memories = {}
        self.environments = {}
        
        # Setup agents
        self._setup_agents()

    def _create_model(self, observation_space, action_space, tl_id):
        """Create Q-network model with advanced configuration support"""
        model_config = self.agent_cfg.get("model", {})
        
        # Basic model parameters
        model_params = {
            "observation_space": observation_space,
            "action_space": action_space,
            "device": self.device,
            "input_dim": observation_space.shape[0],
            "output_dim": action_space.n,
            "num_layers": model_config.get("num_layers", 3),
            "hidden_size": model_config.get("hidden_size", 256),
        }
        
        # Advanced features from original model
        model_params.update({
            "use_attention": model_config.get("use_attention", True),
            "loss_type": model_config.get("loss_type", "mse"),  # mse, huber, weighted, qr
            "num_quantiles": model_config.get("num_quantiles", 100),
            "v_min": model_config.get("v_min", -10.0),
            "v_max": model_config.get("v_max", 10.0),
            "dropout_rate": model_config.get("dropout_rate", 0.0),
            "batch_norm": model_config.get("batch_norm", False),
            "layer_sizes": model_config.get("layer_sizes", None),  # Custom architecture
        })
        
        # Model type selection (for future extensibility)
        model_type = model_config.get("type", "qnetwork")
        
        if model_type == "qnetwork":
            return QNetwork(**model_params).to(self.device)
        elif model_type == "dueling":
            # Future: DuelingQNetwork implementation
            from models.dueling_q_network import DuelingQNetwork
            return DuelingQNetwork(**model_params).to(self.device)
        elif model_type == "noisy":
            # Future: NoisyQNetwork implementation  
            from models.noisy_q_network import NoisyQNetwork
            return NoisyQNetwork(**model_params).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _setup_agents(self):
        """Setup SKRL agents for each traffic light"""
        for tl in self.traffic_lights:
            tl_id = tl["id"]

            # Create environment for this traffic light
            env = TrafficLightEnvironment(self.sim)
            env.set_traffic_light(tl_id, tl)
            self.environments[tl_id] = env

            # Create observation and action spaces
            observation_space_size = self.agent_cfg["num_states"]
            action_space_size = self.sim.num_actions[tl_id]

            # Create proper Gym spaces for SKRL
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_space_size,),
                dtype=np.float32,
            )
            action_space = gym.spaces.Discrete(action_space_size)

            # Create memory with proper tensor names
            tensor_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]
            memory = RandomMemory(
                memory_size=self.agent_cfg.get("memory_size", 10000),
                num_envs=1,
                device=self.device,
            )
            
            # Initialize memory with a dummy sample to set up tensor structure
            dummy_state = torch.zeros((1, observation_space_size), dtype=torch.float32, device=self.device)
            dummy_action = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            dummy_reward = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
            dummy_next_state = torch.zeros((1, observation_space_size), dtype=torch.float32, device=self.device)
            dummy_terminated = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
            dummy_truncated = torch.zeros((1, 1), dtype=torch.bool, device=self.device)
            
            # Add the dummy sample to initialize tensor structure
            memory.add_samples(
                states=dummy_state,
                actions=dummy_action,
                rewards=dummy_reward,
                next_states=dummy_next_state,
                terminated=dummy_terminated,
                truncated=dummy_truncated,
            )
            
            # Reset memory after initialization (removes the dummy sample)
            memory.reset()
            
            self.memories[tl_id] = memory

            # Create Q-network models using the factory method
            q_network = self._create_model(observation_space, action_space, tl_id)
            target_q_network = self._create_model(observation_space, action_space, tl_id)

            # Setup DQN agent configuration
            dqn_cfg = DQN_DEFAULT_CONFIG.copy()
            dqn_cfg.update(
                {
                    "learning_rate": self.agent_cfg.get("model", {}).get("learning_rate", 0.001),
                    "gamma": self.agent_cfg.get("gamma", 0.99),
                    "batch_size": self.agent_cfg.get("model", {}).get("batch_size", 32),
                    "epsilon_initial": 1.0,
                    "epsilon_final": 0.01,
                    "epsilon_decay_episodes": 1000,
                    "target_network_update_freq": self.updating_target_network_steps,
                    "memory_size": self.agent_cfg.get("memory_size", 10000),
                    "learning_starts": 1000,
                }
            )

            # Create DQN agent
            agent = DQN(
                models={"q_network": q_network, "target_q_network": target_q_network},
                memory=memory,
                cfg=dqn_cfg,
                observation_space=observation_space,
                action_space=action_space,
                device=self.device,
            )
            
            # SKRL DQN agents need tensors_names to be set manually
            # This must match what the memory records
            agent.tensors_names = ["states", "actions", "rewards", "next_states", "terminated", "truncated"]

            self.agents[tl_id] = agent

    def select_action(self, tl_id: str, state: np.ndarray, epsilon: float) -> Tuple[float, int]:
        """Select action using SKRL agent"""
        agent = self.agents[tl_id]

        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # For noisy networks, reset noise before action selection
        model_type = self.agent_cfg.get("model", {}).get("type", "qnetwork")
        if model_type == "noisy" and hasattr(agent.models["q_network"], 'reset_noise'):
            agent.models["q_network"].reset_noise()

        # Use epsilon-greedy action selection (for noisy networks, noise provides exploration)
        if model_type == "noisy":
            # Noisy networks don't need epsilon-greedy as noise provides exploration
            with torch.no_grad():
                q_values = agent.models["q_network"].compute({"states": state_tensor})[0]
                action = q_values.argmax().item()
            random_val = 0.0
        else:
            # Standard epsilon-greedy for other models
            if np.random.random() < epsilon:
                action = np.random.randint(0, self.sim.num_actions[tl_id])
                random_val = epsilon
            else:
                with torch.no_grad():
                    q_values = agent.models["q_network"].compute({"states": state_tensor})[0]
                    action = q_values.argmax().item()
                random_val = 0.0

        return random_val, action

    def store_transition(self, tl_id: str, state: np.ndarray, action: int, 
                        reward: float, next_state: np.ndarray, done: bool):
        """Store transition in agent's memory"""
        memory = self.memories[tl_id]

        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)

        # Store in memory
        memory.add_samples(
            states=state_tensor.unsqueeze(0),
            actions=action_tensor.unsqueeze(0),
            rewards=reward_tensor.unsqueeze(0),
            next_states=next_state_tensor.unsqueeze(0),
            terminated=done_tensor.unsqueeze(0),
            truncated=torch.BoolTensor([False]).to(self.device).unsqueeze(0),
        )

    def train_agents(self, step: int, max_steps: int) -> float:
        """Train all agents using SKRL"""
        total_loss = 0.0
        num_trained = 0

        for tl_id, agent in self.agents.items():
            memory = self.memories[tl_id]

            # Check if we have enough samples
            if len(memory) < self.agent_cfg.get("model", {}).get("batch_size", 32):
                continue

            # Train the agent using SKRL's internal training process
            try:
                # SKRL agents handle training internally via post_interaction
                # when they have enough samples
                if (len(memory) >= agent.cfg.get("learning_starts", 100) and 
                    hasattr(agent, 'tensors_names') and agent.tensors_names):
                    agent.post_interaction(timestep=step, timesteps=max_steps)
                    num_trained += 1

            except Exception as e:
                print(f"Training error for agent {tl_id}: {e}")
                continue

        return total_loss / max(num_trained, 1)

    def update_target_networks(self, step: int, max_steps: int):
        """Update target networks for all agents"""
        for tl_id, agent in self.agents.items():
            memory = self.memories[tl_id]
            # Only update if we have enough samples and agent is properly configured
            if (len(memory) >= self.agent_cfg.get("model", {}).get("batch_size", 32) and 
                hasattr(agent, 'tensors_names') and agent.tensors_names):
                try:
                    agent.post_interaction(timestep=step, timesteps=max_steps)
                except Exception as e:
                    print(f"Target network update error for agent {tl_id}: {e}")
                    continue

    def save_models(self, path: str, episode: int = None):
        """Save all SKRL models"""
        for tl_id, agent in self.agents.items():
            if episode is not None:
                model_path = f"{path}/skrl_model_{tl_id}_episode_{episode}.pt"
            else:
                model_path = f"{path.replace('.pt', '')}_{tl_id}.pt"
            torch.save(agent.models["q_network"].state_dict(), model_path)

    def load_models(self, path: str, episode: int = None):
        """Load all SKRL models"""
        for tl_id, agent in self.agents.items():
            if episode is not None:
                model_path = f"{path}/skrl_model_{tl_id}_episode_{episode}.pt"
            else:
                model_path = f"{path.replace('.pt', '')}_{tl_id}.pt"
            try:
                agent.models["q_network"].load_state_dict(torch.load(model_path))
                agent.models["target_q_network"].load_state_dict(torch.load(model_path))
                print(f"Loaded model for {tl_id}")
            except FileNotFoundError:
                print(f"Model file not found: {model_path}")

    def save_checkpoints(self, path: str, episode: int = None, epsilon: float = None):
        """Save all SKRL model checkpoints"""
        for tl_id, agent in self.agents.items():
            checkpoint_path = f"{path.replace('.pt', '')}_{tl_id}_checkpoint.pt"
            checkpoint = {
                "model_state_dict": agent.models["q_network"].state_dict(),
                "target_model_state_dict": agent.models["target_q_network"].state_dict(),
                "episode": episode,
                "epsilon": epsilon,
            }
            torch.save(checkpoint, checkpoint_path)

    @property
    def models(self):
        """Property to access SKRL models for compatibility"""
        return {tl_id: agent.models for tl_id, agent in self.agents.items()}
