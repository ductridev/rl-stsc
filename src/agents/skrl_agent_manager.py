"""
SKRL agent setup and management for traffic light control.
"""

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, List, Tuple, Optional
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

    def __init__(
        self,
        simulation_instance,
        agent_cfg: Dict,
        traffic_lights: List[Dict],
        updating_target_network_steps: int = 100,
        device: torch.device = None,
    ):
        self.sim = simulation_instance
        self.agent_cfg = agent_cfg
        self.traffic_lights = traffic_lights
        self.updating_target_network_steps = updating_target_network_steps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Set random seed for reproducibility
        set_seed(42)

        # Initialize containers
        self.agents: dict[str, DQN] = {}
        self.memories: dict[str, RandomMemory] = {}
        self.environments: dict[str, TrafficLightEnvironment] = {}

        # Setup agents
        self._setup_agents()

        # Additional configuration for batch training
        self._training_interval = simulation_instance.training_steps  # decisions between batch updates
        self._batch_updates = self.agent_cfg.get("batch_updates", 10)  # gradient batches per training interval

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
                memory_size=self.agent_cfg.get("model", {}).get("memory_size", 50000),  # Larger memory for better learning
                num_envs=1,
                device=self.device,
            )

            # Create tensors dictionary and use create_tensor to register the tensor names properly
            print(f"Initializing memory for {tl_id}...")

            # SKRL requires tensors to be registered using create_tensor
            memory.create_tensor(
                name="states", size=observation_space_size, dtype=torch.float32
            )
            memory.create_tensor(name="actions", size=1, dtype=torch.long)
            memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            memory.create_tensor(
                name="next_states", size=observation_space_size, dtype=torch.float32
            )
            memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            print(f"Memory tensors registered: {memory.get_tensor_names()}")

            self.memories[tl_id] = memory

            # Create Q-network models using the factory method
            q_network = self._create_model(observation_space, action_space, tl_id)
            target_q_network = self._create_model(
                observation_space, action_space, tl_id
            )

            # Setup DQN agent configuration
            dqn_cfg = DQN_DEFAULT_CONFIG.copy()

            # Update with our specific configuration
            dqn_cfg.update(
                {
                    "learning_rate": self.agent_cfg.get("model", {}).get(
                        "learning_rate", 0.001
                    ),
                    "discount_factor": self.agent_cfg.get(
                        "gamma", 0.99
                    ),  # Use correct SKRL parameter name
                    "batch_size": self.agent_cfg.get("model", {}).get("batch_size", 256),  # Match config value
                    "learning_starts": 1000,  # Wait for more experience before training
                    "target_update_interval": self.updating_target_network_steps,  # Use correct SKRL parameter name
                    "exploration": {
                        "initial_epsilon": self.agent_cfg.get("model", {}).get(
                            "epsilon", 1
                        ),
                        "final_epsilon": self.agent_cfg.get("model", {}).get(
                            "min_epsilon", 0.001
                        ),
                        "timesteps": 3240000,  # Decay over 900 episodes, stay at 0.05 for episodes 900-1000
                    },
                    "polyak": 0.005,  # Slower target network updates for stability
                    "gradient_steps": 1,  # Single gradient step per update for stability
                    "update_interval": 1,  # Update every timestep (controlled by our batch training)
                    # Ensure experiment settings are properly configured
                    "experiment": {
                        "directory": "",
                        "experiment_name": "",
                        "write_interval": 1200,
                        "checkpoint_interval": 1200,
                        "store_separately": True,
                        "wandb": False,
                        "wandb_kwargs": {},
                    },
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

    def pre_interaction(self, tl_id: str, step: int, max_steps: int):
        agent = self.agents[tl_id]

        agent.pre_interaction(timestep=step, timesteps=max_steps)

    def select_action(
        self, tl_id: str, state: np.ndarray, step: int, max_steps: int
    ) -> int:
        """Select action using SKRL agent"""
        agent = self.agents[tl_id]
        
        # Convert state to tensor with proper shape for SKRL
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # SKRL's DQN agent.act() returns: (tensor([[action]], device='cuda:0'), None, None)
        action_result = agent.act(state_tensor, timestep=step, timesteps=max_steps)
        
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
            # Last resort: try to convert to int
            return int(action_tensor)
    
    def update_step(self, tl_id: str, step: int, max_steps: int):
        """Update step for the agent"""
        agent = self.agents[tl_id]
        if len(agent.memory) > 0:
            agent._update(timestep=step, timesteps=max_steps)

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
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).to(self.device)
        next_state_tensor = torch.FloatTensor(next_state).to(self.device)
        action_tensor = torch.LongTensor([action]).to(self.device)
        reward_tensor = torch.FloatTensor([reward]).to(self.device)
        done_tensor = torch.BoolTensor([done]).to(self.device)

        agent = self.agents[tl_id]
        agent.record_transition(
            states=state_tensor.unsqueeze(0),
            actions=action_tensor.unsqueeze(0),
            rewards=reward_tensor.unsqueeze(0),
            next_states=next_state_tensor.unsqueeze(0),
            terminated=done_tensor.unsqueeze(0),
            truncated=torch.BoolTensor([False]).to(self.device).unsqueeze(0),
            infos=None,
            timestep=step,
            timesteps=max_steps,
        )

    def train_agents(self, tl_id: str, step: int, max_steps: int) -> float:
        """(Legacy) single training trigger using post_interaction once"""
        agent = self.agents[tl_id]
        total_loss = 0

        # Call post_interaction which handles the training step internally
        agent.post_interaction(timestep=step, timesteps=max_steps)

        # Try to extract loss information if available from the agent
        # SKRL DQN agents store their losses in different ways
        if hasattr(agent, "tracking_data") and agent.tracking_data:
            # Get the latest loss from tracking data
            for key, value in agent.tracking_data.items():
                if (
                    "loss" in key.lower()
                    and isinstance(value, (list, tuple))
                    and value
                ):
                    latest_loss = (
                        value[-1]
                        if isinstance(value[-1], (int, float))
                        else (
                            value[-1].item()
                            if hasattr(value[-1], "item")
                            else 0
                        )
                    )
                    total_loss += latest_loss
                    break
        elif hasattr(agent, "_q_loss") and agent._q_loss is not None:
            # Some SKRL agents store loss in _q_loss
            loss_val = (
                agent._q_loss.item()
                if hasattr(agent._q_loss, "item")
                else agent._q_loss
            )
            total_loss += loss_val
        elif hasattr(agent, "_losses") and agent._losses:
            # SKRL agents typically store losses in a dictionary
            for loss_name, loss_value in agent._losses.items():
                if isinstance(loss_value, (int, float)):
                    total_loss += loss_value
                elif hasattr(loss_value, "item"):  # torch tensor
                    total_loss += loss_value.item()

        return total_loss

    def maybe_batch_train(self, tl_id: str, decision_counter: int, max_steps: int):
        """Perform batched training every configured decision interval.
        Args:
            tl_id: traffic light id
            decision_counter: number of decisions (phase selections) so far in episode
            max_steps: max steps (passed for API compatibility)
        """
        if self._training_interval <= 0:
            return 0.0
        if decision_counter == 0 or (decision_counter % self._training_interval) != 0:
            return 0.0
        
        # Check if agent has enough experience to train
        agent = self.agents[tl_id]
        if len(agent.memory) < agent.cfg.get("learning_starts", 1000):
            return 0.0  # Wait for sufficient experience
        agent = self.agents[tl_id]
        total_loss = 0.0
        # Run multiple update cycles (post_interaction) to simulate batch training
        for _ in range(self._batch_updates):
            agent.post_interaction(timestep=decision_counter, timesteps=max_steps)
            # collect loss if available
            if hasattr(agent, "_q_loss") and agent._q_loss is not None:
                try:
                    total_loss += float(agent._q_loss.item())
                except Exception:
                    pass
        return total_loss / max(1, self._batch_updates)

    # ...existing code...
    def save_models(self, path: str, episode: Optional[int] = None):
        """Save all SKRL models"""
        for tl_id, agent in self.agents.items():
            if episode is not None:
                model_path = f"{path}/skrl_model_{tl_id}_episode_{episode}.pt"
            else:
                model_path = f"{path.replace('.pt', '')}_{tl_id}.pt"
            torch.save(agent.models["q_network"].state_dict(), model_path)

    def load_models(self, path: str, episode: Optional[int] = None):
        """Load all SKRL models with improved file detection"""
        import glob
        
        loaded_count = 0
        for tl_id, agent in self.agents.items():
            model_loaded = False
            
            # Try different loading strategies
            try:
                if episode is not None:
                    if episode == "final":
                        # Try to load final episode model
                        model_patterns = [
                            f"{path}/skrl_model_{tl_id}_episode_final.pt",
                            f"{path}/skrl_model_{tl_id}_FINAL.pt"
                        ]
                    else:
                        # Try to load specific episode or BEST model
                        model_patterns = [
                            f"{path}/skrl_model_{tl_id}_episode_{episode}.pt",
                            f"{path}/skrl_model_{tl_id}_episode_{episode}_BEST.pt"
                        ]
                else:
                    # Try to load in order of preference
                    model_patterns = [
                        f"{path}/skrl_model_{tl_id}_GLOBAL_BEST.pt",
                        f"{path}/skrl_model_{tl_id}_CURRENT_BEST.pt",
                        f"{path.rstrip('/')}/skrl_model_{tl_id}_GLOBAL_BEST.pt",
                        f"{path.rstrip('/')}/skrl_model_{tl_id}_CURRENT_BEST.pt"
                    ]
                    
                    # Also try to find any BEST models with episode numbers
                    best_files = glob.glob(f"{path.rstrip('/')}/skrl_model_{tl_id}_episode_*_BEST.pt")
                    if best_files:
                        # Sort by episode number (extract from filename)
                        try:
                            best_files.sort(key=lambda x: int(x.split("_episode_")[1].split("_")[0]), reverse=True)
                            model_patterns.insert(0, best_files[0])  # Add latest BEST model at the front
                        except (ValueError, IndexError):
                            model_patterns.extend(best_files)
                    
                    # Finally, try any model for this traffic light
                    any_models = glob.glob(f"{path.rstrip('/')}/skrl_model_{tl_id}_*.pt")
                    if any_models:
                        model_patterns.extend(any_models)
                
                # Try each pattern until one works
                for model_path in model_patterns:
                    if os.path.exists(model_path):
                        try:
                            # Load the state dict
                            state_dict = torch.load(model_path, map_location=self.device)
                            agent.models["q_network"].load_state_dict(state_dict)
                            agent.models["target_q_network"].load_state_dict(state_dict)
                            
                            print(f"Loaded model for {tl_id} from: {os.path.basename(model_path)}")
                            model_loaded = True
                            loaded_count += 1
                            break
                        except Exception as e:
                            print(f"Failed to load {model_path}: {e}")
                            continue
                
                if not model_loaded:
                    print(f"No suitable model found for {tl_id} in {path}")
                    # List available files for debugging
                    available_files = glob.glob(f"{path.rstrip('/')}/skrl_model_{tl_id}_*.pt")
                    if available_files:
                        print(f"   Available files: {[os.path.basename(f) for f in available_files]}")
                    else:
                        print(f"   No .pt files found for {tl_id}")
                        
            except Exception as e:
                print(f"Error loading model for {tl_id}: {e}")
        
        if loaded_count > 0:
            print(f"Successfully loaded models for {loaded_count}/{len(self.agents)} traffic lights")
        else:
            raise FileNotFoundError(f"Failed to load any models from {path}")
        
        return loaded_count

    def save_checkpoints(
        self, path: str, episode: Optional[int] = None
    ):
        """Save all SKRL model checkpoints"""
        for tl_id, agent in self.agents.items():
            checkpoint_path = f"{path.replace('.pt', '')}_{tl_id}_checkpoint.pt"
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
