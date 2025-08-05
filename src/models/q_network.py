"""
Q-Network model for DQN using SKRL with advanced features from the original model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skrl.models.torch import Model, DeterministicMixin
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


class QNetwork(DeterministicMixin, Model):
    """Advanced Q-network model for DQN using SKRL with features from original model"""

    def __init__(self, observation_space, action_space, device="cpu", **kwargs):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self)

        # Basic configuration
        self.input_dim = kwargs.get("input_dim", observation_space.shape[0])
        self.output_dim = kwargs.get("output_dim", action_space.n)
        self.device = device
        
        # Architecture configuration
        self.num_layers = kwargs.get("num_layers", 3)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.use_attention = kwargs.get("use_attention", True)
        
        # Advanced features from original model
        self.loss_type = kwargs.get("loss_type", "mse")  # mse, huber, weighted, qr
        self.num_quantiles = kwargs.get("num_quantiles", 100)
        self.v_min = kwargs.get("v_min", -10.0)
        self.v_max = kwargs.get("v_max", 10.0)
        
        # Regularization
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        self.batch_norm = kwargs.get("batch_norm", False)
        
        # Custom layer sizes (if provided, overrides num_layers and hidden_size)
        self.layer_sizes = kwargs.get("layer_sizes", None)
        
        # Build the network components
        self.backbone = self._build_backbone()
        self.attention = self._build_attention() if self.use_attention else None
        self.head = self._build_head()

    def _build_backbone(self):
        """Build the backbone network (feature extractor)"""
        layers = []
        
        if self.layer_sizes is not None:
            # Use custom layer architecture
            layer_sizes = [self.input_dim] + self.layer_sizes
            
            for i in range(len(layer_sizes) - 1):
                layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                
                if self.batch_norm:
                    layers.append(nn.LayerNorm(layer_sizes[i + 1]))
                
                layers.append(nn.ReLU())
                
                if self.dropout_rate > 0:
                    layers.append(nn.Dropout(self.dropout_rate))
            
            final_size = layer_sizes[-1]
        else:
            # Use default architecture similar to original model
            # Input layer
            layers.append(nn.Linear(self.input_dim, 128))
            if self.batch_norm:
                layers.append(nn.LayerNorm(128))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            # Second layer
            layers.append(nn.Linear(128, self.hidden_size))
            if self.batch_norm:
                layers.append(nn.LayerNorm(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            # Additional hidden layers
            for _ in range(self.num_layers):
                layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                if self.batch_norm:
                    layers.append(nn.LayerNorm(self.hidden_size))
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

    def _build_head(self):
        """Build the output head based on loss type"""
        if self.loss_type in ("qr", "wasserstein"):
            # Quantile regression: output num_quantiles for each action
            return nn.Linear(self.final_feature_size, self.output_dim * self.num_quantiles)
        else:
            # Standard Q-values: one output per action
            return nn.Linear(self.final_feature_size, self.output_dim)

    def compute(self, inputs, role=""):
        """Compute Q-values for given inputs (SKRL interface)"""
        states = inputs["states"]
        q_values = self._forward_internal(states)
        return q_values, {}
        
    def _forward_internal(self, x):
        """Internal forward pass through the network"""
        # Backbone feature extraction
        features = self.backbone(x)  # [B, feature_size]
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Output head
        output = self.head(features)
        
        # Reshape for quantile regression
        if self.loss_type in ("qr", "wasserstein"):
            batch_size = output.shape[0]
            output = output.view(batch_size, self.output_dim, self.num_quantiles)
        
        return output
        
    def act(self, inputs, role=""):
        """Get Q-values for given inputs (SKRL interface)"""
        q_values = self._forward_internal(inputs["states"])
        
        if self.loss_type in ("qr", "wasserstein"):
            # For quantile regression, average across quantiles to get Q-values
            q_values = q_values.mean(dim=-1)
        
        # SKRL expects Q-values, not actions!
        return q_values, None, {}

    def predict_one(self, x):
        """Predict Q-values for a single input (compatibility method)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(np.reshape(x, [1, self.input_dim]), dtype=torch.float32).to(self.device)
        else:
            x = torch.reshape(x, [1, self.input_dim])
        return self._forward_internal(x)

    def predict_batch(self, x):
        """Predict Q-values for a batch of inputs (compatibility method)"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return self._forward_internal(x)

    def get_q_values(self, states):
        """Get Q-values for given states (handles different loss types)"""
        output = self._forward_internal(states)
        
        if self.loss_type in ("qr", "wasserstein"):
            # Average across quantiles for standard Q-values
            return output.mean(dim=-1)
        else:
            return output

    def summary(self, input_size=None):
        """Print model summary"""
        if input_size is None:
            input_size = (1, self.input_dim)
        
        print("=== Q-Network Architecture Summary ===")
        print(f"Loss type: {self.loss_type}")
        print(f"Input dim: {self.input_dim}")
        print(f"Output dim: {self.output_dim}")
        print(f"Use attention: {self.use_attention}")
        
        if self.loss_type in ("qr", "wasserstein"):
            print(f"Num quantiles: {self.num_quantiles}")
            print(f"Total output size: {self.output_dim * self.num_quantiles}")
        
        print("\nBackbone:")
        summary(self.backbone, input_size=input_size)
        
        if self.attention is not None:
            print("\nAttention:")
            summary(self.attention, input_size=(input_size[0], self.final_feature_size))
        
        print("\nHead:")
        summary(self.head, input_size=(input_size[0], self.final_feature_size))
        print("=" * 40)
