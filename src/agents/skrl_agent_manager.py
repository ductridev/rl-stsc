"""
SKRL agent setup and management for traffic light control.
"""

import torch
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
from memory_palace import MemoryPalace
from skrl.memories.torch import RandomMemory
from skrl.agents.torch.dqn import DQN_DEFAULT_CONFIG
import sys
import os

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from models.q_network import QNetwork, MLP
from environment.traffic_light_env import TrafficLightEnvironment
from skrl.resources.preprocessors.torch import RunningStandardScaler
# from agents.custom_dqn import CustomDQN
# from skrl.agents.torch.dqn import DQN, DDQN
from agents.custom_dqn import DQN

class SKRLAgentManager:
    """Manager for SKRL agents and components"""

    def __init__(
        self,
        simulation_instance,
        agent_cfg: Dict,
        traffic_lights: List[Dict],
        updating_target_network_steps: int = 100,
        device: torch.device = None,
        evaluation_mode: bool = False,
    ):
        self.sim = simulation_instance
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.updating_target_network_steps = updating_target_network_steps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.evaluation_mode = evaluation_mode

        # Initialize containers
        self.agents: dict[str, DQN] = {}
        self.memories: dict[str, MemoryPalace] = {}
        self.environments: dict[str, TrafficLightEnvironment] = {}

        # Setup agents
        self._setup_agents()

    def _create_model(self, observation_space, action_space, tl_id):
        """Create Q-network model with advanced configuration support"""
        model_config = self.agent_cfg.get("model", {})

        # Basic model parameters
        model_params = {
            "observation_space": observation_space,
            "action_space": action_space,
            "device": str(self.device),  # Convert device to string for QNetwork
            "input_dim": observation_space.shape[0],
            "output_dim": action_space.n,
            "num_layers": model_config.get("num_layers", 3),
            "hidden_size": model_config.get("hidden_size", 256),
        }

        # Advanced features from original model
        model_params.update(
            {
                "use_attention": model_config.get("use_attention", True),
                "loss_type": model_config.get(
                    "loss_type", "mse"
                ),  # mse, huber, weighted, qr
                "num_quantiles": model_config.get("num_quantiles", 100),
                "v_min": model_config.get("v_min", -10.0),
                "v_max": model_config.get("v_max", 10.0),
                "dropout_rate": model_config.get("dropout_rate", 0.0),
                "batch_norm": model_config.get("batch_norm", False),
                "layer_sizes": model_config.get(
                    "layer_sizes", None
                ),  # Custom architecture
            }
        )

        # Model type selection (for future extensibility)
        model_type = model_config.get("type", "qnetwork")

        # return MLP(observation_space=observation_space,
        #      action_space=action_space,
        #      device=self.device,
        #      clip_actions=False).to(self.device)

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
            tensor_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]
            memory = RandomMemory(
                memory_size= 10000 if self.evaluation_mode else self.sim.memory_size[1],
                num_envs=1,
                device=self.device,
            )

            self.memories[tl_id] = memory

            # Create Q-network models using the factory method
            q_network = self._create_model(observation_space, action_space, tl_id)
            target_q_network = self._create_model(
                observation_space, action_space, tl_id
            )

            q_network.summary()

            # Setup DQN agent configuration
            dqn_cfg = DQN_DEFAULT_CONFIG.copy()

            # Update with our specific configuration
            dqn_cfg.update(
                {
                    "learning_rate": min(self.agent_cfg.get("model", {}).get(
                        "learning_rate", 0.0001
                    ), 0.0001),  # Lower learning rate to prevent instability
                    "learning_rate_scheduler": KLAdaptiveLR,
                    "learning_rate_scheduler_kwargs": {
                        "kl_threshold": 0.01,
                    },
                    # "state_preprocessor": RunningStandardScaler,
                    # "state_preprocessor_kwargs": {
                    #     "size": observation_space.shape[0], 
                    #     "device": self.device
                    # },
                    "mixed_precision": True,  # Disable mixed precision to prevent NaN issues
                    "discount_factor": self.agent_cfg.get(
                        "gamma", 0.99
                    ),  # Use correct SKRL parameter name
                    "batch_size": self.agent_cfg.get("model", {}).get(
                        "batch_size", 64
                    ),  # Match config value
                    "random_timesteps": self.sim.max_steps,
                    "learning_starts": self.agent_cfg.get("model", {}).get(
                        "batch_size", 64
                    ) * 5,  # Wait for more experience before training
                    "target_update_interval": self.updating_target_network_steps,  # Use correct SKRL parameter name
                    "exploration": {
                        "initial_epsilon": 0.0 if self.evaluation_mode else self.agent_cfg.get("model", {}).get(
                            "initial_epsilon", 0.95
                        ),
                        "final_epsilon": 0.0 if self.evaluation_mode else self.agent_cfg.get("model", {}).get(
                            "min_epsilon", 0.01
                        ),
                        "timesteps": 0 if self.evaluation_mode else self.agent_cfg.get("model", {}).get(
                            "decay_episode", 90
                        )
                        * self.sim.max_steps,  # Decay over 90 episodes, stay at min for episodes 90-100
                    },
                    "polyak": 1,  # Slower target network updates for stability
                    "gradient_steps": 0 if self.evaluation_mode else 500,  # No gradient steps in evaluation mode
                    "update_interval": 0 if self.evaluation_mode else self.sim.max_steps,  # No updates in evaluation mode
                }
            )

            # Remove any conflicting legacy parameters
            legacy_params = [
                "epsilon_initial",
                "epsilon_final",
                "epsilon_decay_episodes",
                "target_network_update_freq",
                "memory_size",
                "gradient_clipping",
                "gamma",
            ]
            for param in legacy_params:
                if param in dqn_cfg:
                    del dqn_cfg[param]

            # Create DQN agent
            agent = DQN(
                models={"q_network": q_network, "target_q_network": target_q_network},
                memory=memory,
                cfg=dqn_cfg,
                observation_space=observation_space,
                action_space=action_space,
                device=self.device,
            )

            agent.init()

            # Set models to evaluation mode if in evaluation mode
            if self.evaluation_mode:
                agent.models["q_network"].eval()
                agent.models["target_q_network"].eval()
                print(f"Agent {tl_id} set to evaluation mode (no training, epsilon=0)")

            print(f"Memory tensors registered: {memory.get_tensor_names()}")

            # SKRL DQN agents need tensors_names to be set manually
            # This must match what the memory records
            agent.tensors_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]

            self.agents[tl_id] = agent

    def post_interaction(self, tl_id: str, step: int, max_steps: int):
        agent = self.agents[tl_id]

        agent.post_interaction(timestep=step, timesteps=max_steps)

    def pre_interaction(self, tl_id: str, step: int, max_steps: int):
        agent = self.agents[tl_id]

        agent.pre_interaction(timestep=step, timesteps=max_steps)

    def select_action(
        self, tl_id: str, state: np.ndarray, step: int, max_steps: int, desra_phase_idx: int
    ) -> int:
        """Select action using SKRL agent"""
        agent = self.agents[tl_id]

        # Convert state to tensor with proper shape for SKRL
        state_tensor = torch.Tensor(state).unsqueeze(0).to(self.device)

        # SKRL's DQN agent.act() returns: (tensor([[action]], device='cuda:0'), None, None)
        if self.evaluation_mode:
            with torch.no_grad():
                action_result = agent.act(state_tensor, step, max_steps, desra_phase_idx)
        else:
            action_result = agent.act(state_tensor, step, max_steps, desra_phase_idx)

        # Extract action from the tuple - first element is the action tensor
        if isinstance(action_result, tuple) and len(action_result) >= 1:
            action_tensor = action_result[0]  # Get tensor([[0]], device='cuda:0')
        else:
            action_tensor = action_result

        # Convert tensor to int - handle 2D tensor case
        if isinstance(action_tensor, torch.Tensor):
            # For tensor([[0]]) format, we need [0][0] to get the scalar
            if action_tensor.dim() == 2:  # 2D tensor like [[0]]
                return action_tensor[0][0].item()
            elif action_tensor.dim() == 1:  # 1D tensor like [0]
                return action_tensor[0].item()
            elif action_tensor.dim() == 0:  # Scalar tensor
                return action_tensor.item()
            else:  # Multi-dimensional tensor
                return action_tensor.flatten()[0].item()
        elif isinstance(action_tensor, (int, float, np.integer, np.floating)):
            return int(action_tensor)
        else:
            # Last resort: try to convert to int
            return int(action_tensor)

    def store_transition(
        self,
        tl_id: str,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        step: int,
        max_steps: int,
    ):
        """Store transition in agent's memory"""
        # Convert to tensors with correct data types
        state_tensor = torch.LongTensor(state).to(self.device)
        next_state_tensor = torch.LongTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.LongTensor([reward]).to(self.device)
        # Ensure terminated and truncated are boolean tensors
        terminated_tensor = torch.tensor([done], dtype=torch.bool, device=self.device)
        truncated_tensor = torch.tensor([False], dtype=torch.bool, device=self.device)

        agent = self.agents[tl_id]
        agent.record_transition(
            states=state_tensor.unsqueeze(0),
            actions=action_tensor.unsqueeze(0),
            rewards=reward_tensor.unsqueeze(0),
            next_states=next_state_tensor.unsqueeze(0),
            terminated=terminated_tensor.unsqueeze(0),
            truncated=truncated_tensor.unsqueeze(0),
            infos=None,
            timestep=step,
            timesteps=max_steps,
        )

    def save_models(self, path: str, episode: Optional[int] = None):
        """Save all SKRL models"""
        for tl_id, agent in self.agents.items():
            if episode is not None:
                model_path = f"{path}/skrl_model_{tl_id}_episode_{episode}.pt"
            else:
                model_path = f"{path.replace('.pt', '')}_{tl_id}.pt"
            torch.save(agent.models["q_network"].state_dict(), model_path)

    def load_models(self, path: str, episode: Optional[int] = None):
        """Load all SKRL models - if path ends with .pt, load directly, otherwise search for models"""
        import glob

        loaded_count = 0
        
        # If path ends with .pt, it's a direct file path
        if path.endswith('.pt'):
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load the same model for all traffic lights
            for tl_id, agent in self.agents.items():
                try:
                    checkpoint = torch.load(path)
                    
                    # Check if it's a checkpoint file (contains dictionary) or direct state_dict
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # It's a checkpoint file saved by save_checkpoints
                        agent.models["q_network"].load_state_dict(checkpoint["model_state_dict"])
                        if "target_model_state_dict" in checkpoint:
                            agent.models["target_q_network"].load_state_dict(checkpoint["target_model_state_dict"])
                        else:
                            # Use same weights for target network if not saved separately
                            agent.models["target_q_network"].load_state_dict(checkpoint["model_state_dict"])
                        print(f"Loaded checkpoint for {tl_id} from: {os.path.basename(path)} (episode: {checkpoint.get('episode', 'unknown')})")
                    else:
                        # It's a direct state_dict file
                        agent.models["q_network"].load_state_dict(checkpoint)
                        agent.models["target_q_network"].load_state_dict(checkpoint)
                        print(f"Loaded model for {tl_id} from: {os.path.basename(path)}")
                    loaded_count += 1
                except Exception as e:
                    print(f"Failed to load {path} for {tl_id}: {e}")
        else:
            # Search for models in the directory
            for tl_id, agent in self.agents.items():
                model_loaded = False
                
                # Search for any .pt model file for this traffic light
                model_patterns = glob.glob(f"{path.rstrip('/')}/skrl_model_{tl_id}_*.pt")
                # Also search for checkpoint files
                checkpoint_patterns = glob.glob(f"{path.rstrip('/')}/_{tl_id}_*_checkpoint.pt")
                model_patterns.extend(checkpoint_patterns)
                
                if model_patterns:
                    # Try to load the first available model
                    for model_path in model_patterns:
                        try:
                            checkpoint = torch.load(model_path)
                            
                            # Check if it's a checkpoint file or direct state_dict
                            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                                # It's a checkpoint file
                                agent.models["q_network"].load_state_dict(checkpoint["model_state_dict"])
                                if "target_model_state_dict" in checkpoint:
                                    agent.models["target_q_network"].load_state_dict(checkpoint["target_model_state_dict"])
                                else:
                                    agent.models["target_q_network"].load_state_dict(checkpoint["model_state_dict"])
                                print(f"Loaded checkpoint for {tl_id} from: {os.path.basename(model_path)} (episode: {checkpoint.get('episode', 'unknown')})")
                            else:
                                # It's a direct state_dict
                                agent.models["q_network"].load_state_dict(checkpoint)
                                agent.models["target_q_network"].load_state_dict(checkpoint)
                                print(f"Loaded model for {tl_id} from: {os.path.basename(model_path)}")
                            
                            model_loaded = True
                            loaded_count += 1
                            break
                        except Exception as e:
                            print(f"Failed to load {model_path}: {e}")
                            continue
                
                if not model_loaded:
                    print(f"No suitable model found for {tl_id} in {path}")
                    available_files = glob.glob(f"{path.rstrip('/')}/skrl_model_{tl_id}_*.pt")
                    checkpoint_files = glob.glob(f"{path.rstrip('/')}/_{tl_id}_*_checkpoint.pt")
                    all_files = available_files + checkpoint_files
                    if all_files:
                        print(f"   Available files: {[os.path.basename(f) for f in all_files]}")
                    else:
                        print(f"   No model files found for {tl_id}")

        if loaded_count > 0:
            print(f"Successfully loaded models for {loaded_count}/{len(self.agents)} traffic lights")
        else:
            raise FileNotFoundError(f"Failed to load any models from {path}")

        return loaded_count

    def save_checkpoints(self, path: str, episode: Optional[int] = None):
        """Save all SKRL model checkpoints"""
        for tl_id, agent in self.agents.items():
            checkpoint_path = f"{path.replace('.pt', '')}_{tl_id}_{episode}_checkpoint.pt"
            checkpoint = {
                "model_state_dict": agent.models["q_network"].state_dict(),
                "target_model_state_dict": agent.models[
                    "target_q_network"
                ].state_dict(),
                "episode": episode,
            }
            torch.save(checkpoint, checkpoint_path)

    @property
    def models(self):
        """Property to access SKRL models for compatibility"""
        return {tl_id: agent.models for tl_id, agent in self.agents.items()}

    def clear_memories(self):
        """Clear all agent memories to prevent accumulation between episodes"""
        print("Clearing agent memories to prevent memory leaks...")
        for tl_id, agent in self.agents.items():
            # SKRL memory doesn't have a direct clear method, but we can reset the internal pointer
            agent.memory.reset()
        print("Agent memories cleared")
