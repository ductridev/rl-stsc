"""
Custom SKRL DQN agent that supports configurable loss functions.
"""

import torch
import torch.nn.functional as F
from skrl.agents.torch.dqn import DQN
from typing import Union, Tuple, Dict, Any


class CustomDQN(DQN):
    """Custom DQN agent that supports configurable loss functions through the Q-network"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Check if the Q-network has a custom loss function
        self.use_custom_loss = hasattr(self.models["q_network"], "compute_loss")
        
        if self.use_custom_loss:
            print(f"Using custom loss function: {self.models['q_network'].loss_type}")
        else:
            print("Using default SKRL MSE loss function")
            
        # Ensure optimizer is properly initialized for Q-network
        if not hasattr(self, 'optimizer'):
            self.optimizer = torch.optim.Adam(
                self.models["q_network"].parameters(), 
                lr=self.cfg.get("learning_rate", 0.001)
            )
    
    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step with custom loss support"""
        
        # Sample a batch from memory  
        sampled_data = self.memory.sample(self.cfg["batch_size"])
        
        # Handle different memory formats
        if isinstance(sampled_data, tuple) and len(sampled_data) > 1:
            # Standard format: (data_dict, indices, weights)
            sampled_tensors = sampled_data[0]
        else:
            # Direct tensor format
            sampled_tensors = sampled_data
            
        # Extract tensors from the sampled data
        sampled_states = sampled_tensors["states"].float()
        sampled_actions = sampled_tensors["actions"].long()
        sampled_rewards = sampled_tensors["rewards"].float()
        sampled_next_states = sampled_tensors["next_states"].float()
        sampled_dones = sampled_tensors["terminated"].bool()

        # Compute Q-values for current states
        q_values, _ = self.models["q_network"].act({"states": sampled_states})
        q_values = q_values.gather(1, sampled_actions)

        # Compute target Q-values
        with torch.no_grad():
            next_q_values, _ = self.models["target_q_network"].act({"states": sampled_next_states})
            target_q_values = sampled_rewards + self.cfg["discount_factor"] * \
                             next_q_values.max(1)[0].unsqueeze(1) * ~sampled_dones

        # Compute loss using custom or default method
        if self.use_custom_loss:
            # Use the Q-network's custom loss function
            loss = self.models["q_network"].compute_loss(q_values, target_q_values)
        else:
            # Use default MSE loss
            loss = F.mse_loss(q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        if self.cfg.get("gradient_clipping", None) is not None:
            torch.nn.utils.clip_grad_norm_(self.models["q_network"].parameters(), 
                                         self.cfg["gradient_clipping"])
        self.optimizer.step()

        # Update tracking variables
        if hasattr(self, 'tracking_data'):
            self.tracking_data["Loss / Q-network loss"] = loss.item()

        # Store loss for external access
        self._q_loss = loss
