"""
Dueling Q-Network model for DQN using SKRL with advanced features.
Dueling DQN separates state value and advantage functions for better learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skrl.models.torch import Model
from torchinfo import summary
from typing import Optional

# Import SENet if available, otherwise create a simple attention module
try:
    from src.SENet_module import SENet
except ImportError:
    class SENet(nn.Module):
        """Simple attention module as fallback"""
        def __init__(self, channel):
            super().__init__()
            self.channel = channel
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channel, channel // 16, 1),
                nn.ReLU(),
                nn.Conv1d(channel // 16, channel, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            # Simple channel attention
            b, c = x.shape
            attn = self.attention(x.unsqueeze(-1)).squeeze(-1)
            return x * attn


class DuelingQNetwork(Model):
    """Dueling Q-network model that separates value and advantage functions"""

    def __init__(self, observation_space, action_space, device="cpu", **kwargs):
        Model.__init__(self, observation_space, action_space, device)

        # Basic configuration
        self.input_dim = kwargs.get("input_dim", observation_space.shape[0])
        self.output_dim = kwargs.get("output_dim", action_space.n)
        self.device = device
        
        # Architecture configuration
        self.num_layers = kwargs.get("num_layers", 3)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.use_attention = kwargs.get("use_attention", True)
        
        # Advanced features
        self.loss_type = kwargs.get("loss_type", "mse")
        self.num_quantiles = kwargs.get("num_quantiles", 100)
        self.v_min = kwargs.get("v_min", -10.0)
        self.v_max = kwargs.get("v_max", 10.0)
        
        # Regularization
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.batch_norm = kwargs.get("batch_norm", False)
        
        # Custom layer sizes
        self.layer_sizes = kwargs.get("layer_sizes", None)
        
        # Build the network components
        self.backbone = self._build_backbone()
        self.attention = self._build_attention() if self.use_attention else None
        
        # Dueling architecture: separate value and advantage streams
        self.value_stream = self._build_value_stream()
        self.advantage_stream = self._build_advantage_stream()

    def _build_backbone(self):
        """Build the shared backbone network (feature extractor)"""
        layers = []
        
        if self.layer_sizes is not None:
            # Use custom layer architecture
            layer_sizes = [self.input_dim] + self.layer_sizes
            
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                
                layers.append(nn.ReLU())
                
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
            
            final_size = layer_sizes[-1]
        else:
            # Use default architecture
            layers.append(nn.Linear(self.input_dim, 128))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(128))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            layers.append(nn.Linear(128, self.hidden_size))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            for _ in range(self.num_layers):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                if self.batch_norm:
                    layers.append(nn.BatchNorm1d(self.hidden_size))
                layers.append(nn.ReLU())
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
            
            final_size = self.hidden_size
        
        self.final_feature_size = final_size
        return nn.Sequential(*layers)

    def _build_attention(self):
        """Build attention mechanism"""
        if self.use_attention:
            return SENet(channel=self.final_feature_size)
        return None

    def _build_value_stream(self):
        """Build value stream that outputs state value V(s)"""
        layers = []
        layers.append(nn.Linear(self.final_feature_size, self.final_feature_size // 2))
        layers.append(nn.ReLU())
        
        if self.loss_type in ("qr", "wasserstein"):
            # For quantile regression: output quantiles for state value
            layers.append(nn.Linear(self.final_feature_size // 2, self.num_quantiles))
        else:
            # Standard value function: single scalar output
            layers.append(nn.Linear(self.final_feature_size // 2, 1))
        
        return nn.Sequential(*layers)

    def _build_advantage_stream(self):
        """Build advantage stream that outputs advantage A(s,a)"""
        layers = []
        layers.append(nn.Linear(self.final_feature_size, self.final_feature_size // 2))
        layers.append(nn.ReLU())
        
        if self.loss_type in ("qr", "wasserstein"):
            # For quantile regression: output quantiles for each action advantage
            layers.append(nn.Linear(self.final_feature_size // 2, self.output_dim * self.num_quantiles))
        else:
            # Standard advantage function: one output per action
            layers.append(nn.Linear(self.final_feature_size // 2, self.output_dim))
        
        return nn.Sequential(*layers)

    def compute(self, inputs, role=""):
        """Forward pass through the network (SKRL interface)"""
        x = inputs["states"]
        return self.forward(x), {}
        
    def forward(self, x):
        """Forward pass through the dueling network"""
        # Shared backbone feature extraction
        features = self.backbone(x)  # [B, feature_size]
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Separate value and advantage streams
        value = self.value_stream(features)  # [B, 1] or [B, num_quantiles]
        advantage = self.advantage_stream(features)  # [B, num_actions] or [B, num_actions * num_quantiles]
        
        # Combine value and advantage to get Q-values
        if self.loss_type in ("qr", "wasserstein"):
            # Reshape for quantile regression
            batch_size = advantage.shape[0]
            advantage = advantage.view(batch_size, self.output_dim, self.num_quantiles)
            value = value.unsqueeze(1).expand(-1, self.output_dim, -1)  # [B, num_actions, num_quantiles]
            
            # Dueling equation: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            advantage_mean = advantage.mean(dim=1, keepdim=True)  # [B, 1, num_quantiles]
            q_values = value + advantage - advantage_mean  # [B, num_actions, num_quantiles]
        else:
            # Standard dueling equation: Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
            advantage_mean = advantage.mean(dim=1, keepdim=True)  # [B, 1]
            q_values = value + advantage - advantage_mean  # [B, num_actions]
        
        return q_values
        
    def act(self, inputs, role=""):
        """Select actions based on Q-values (SKRL interface)"""
        q_values = self.forward(inputs["states"])
        
        if self.loss_type in ("qr", "wasserstein"):
            # For quantile regression, average across quantiles to get Q-values
            q_values = q_values.mean(dim=-1)
        
        # Select action with highest Q-value
        actions = torch.argmax(q_values, dim=-1, keepdim=True)
        return actions, None, {}

    def predict_one(self, x):
        """Predict Q-values for a single input (compatibility method)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.reshape(x, [1, self.input_dim]), dtype=torch.float32).to(self.device)
        else:
            x = torch.reshape(x, [1, self.input_dim])
        return self.forward(x)

    def predict_batch(self, x):
        """Predict Q-values for a batch of inputs (compatibility method)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return self.forward(x)

    def get_q_values(self, states):
        """Get Q-values for given states (handles different loss types)"""
        output = self.forward(states)
        
        if self.loss_type in ("qr", "wasserstein"):
            # Average across quantiles for standard Q-values
            return output.mean(dim=-1)
        else:
            return output

    def get_value_and_advantage(self, states):
        """Get separate value and advantage estimates (useful for analysis)"""
        features = self.backbone(states)
        
        if self.attention is not None:
            features = self.attention(features)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        if self.loss_type in ("qr", "wasserstein"):
            batch_size = advantage.shape[0]
            advantage = advantage.view(batch_size, self.output_dim, self.num_quantiles)
            # For quantile regression, return mean across quantiles for analysis
            value = value.mean(dim=-1, keepdim=True)
            advantage = advantage.mean(dim=-1)
        
        return value, advantage

    def summary(self, input_size=None):
        """Print model summary"""
        if input_size is None:
            input_size = (1, self.input_dim)
        
        print("=== Dueling Q-Network Architecture Summary ===")
        print(f"Loss type: {self.loss_type}")
        print(f"Input dim: {self.input_dim}")
        print(f"Output dim: {self.output_dim}")
        print(f"Use attention: {self.use_attention}")
        
        if self.loss_type in ("qr", "wasserstein"):
            print(f"Num quantiles: {self.num_quantiles}")
        
        print("\nShared Backbone:")
        summary(self.backbone, input_size=input_size)
        
        if self.attention is not None:
            print("\nAttention:")
            summary(self.attention, input_size=(input_size[0], self.final_feature_size))
        
        print("\nValue Stream:")
        summary(self.value_stream, input_size=(input_size[0], self.final_feature_size))
        
        print("\nAdvantage Stream:")
        summary(self.advantage_stream, input_size=(input_size[0], self.final_feature_size))
        print("=" * 50)
