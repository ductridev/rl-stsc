"""
Noisy Q-Network model for DQN using SKRL with advanced features.
Noisy Networks use parametric noise for exploration instead of epsilon-greedy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skrl.models.torch import Model
from torchinfo import summary
from typing import Optional
import math

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


class NoisyLinear(nn.Module):
    """Noisy linear layer with factorized Gaussian noise"""
    
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        # Noise tensors (not parameters)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialize parameters"""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Reset noise tensors"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size):
        """Scale noise using factorized Gaussian noise"""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def forward(self, input):
        """Forward pass with noisy weights"""
        if self.training:
            # Use noisy weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use deterministic weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class NoisyQNetwork(Model):
    """Noisy Q-network model that uses parametric noise for exploration"""

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
        
        # Noisy network specific parameters
        self.std_init = kwargs.get("noise_std", 0.5)  # Initial noise standard deviation
        self.noisy_layers_only = kwargs.get("noisy_layers_only", True)  # Only make final layers noisy
        
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
        self.head = self._build_head()

    def _build_backbone(self):
        """Build the backbone network (feature extractor)"""
        layers = []
        
        if self.layer_sizes is not None:
            # Use custom layer architecture
            layer_sizes = [self.input_dim] + self.layer_sizes
            
            for i in range(len(layer_sizes) - 1):
                # Use regular linear layers for backbone (noise only in head if specified)
                if self.noisy_layers_only or i < len(layer_sizes) - 2:
                    layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
                else:
                    layers.append(NoisyLinear(layer_sizes[i], layer_sizes[i + 1], self.std_init))
                
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
            
            # Add hidden layers (use noisy layers for the last few if not noisy_layers_only)
            for i in range(self.num_layers):
                if self.noisy_layers_only or i < self.num_layers - 1:
                    layers.append(nn.Linear(self.hidden_size, self.hidden_size))
                else:
                    layers.append(NoisyLinear(self.hidden_size, self.hidden_size, self.std_init))
                
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

    def _build_head(self):
        """Build the output head with noisy layers"""
        if self.loss_type in ("qr", "wasserstein"):
            # Quantile regression: output num_quantiles for each action
            return NoisyLinear(self.final_feature_size, self.output_dim * self.num_quantiles, self.std_init)
        else:
            # Standard Q-values: one output per action
            return NoisyLinear(self.final_feature_size, self.output_dim, self.std_init)

    def reset_noise(self):
        """Reset noise in all noisy layers"""
        def _reset_noise(module):
            if isinstance(module, NoisyLinear):
                module.reset_noise()
        
        self.apply(_reset_noise)

    def compute(self, inputs, role=""):
        """Forward pass through the network (SKRL interface)"""
        x = inputs["states"]
        return self.forward(x), {}
        
    def forward(self, x):
        """Forward pass through the network"""
        # Backbone feature extraction
        features = self.backbone(x)  # [B, feature_size]
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Output head (always noisy)
        output = self.head(features)
        
        # Reshape for quantile regression
        if self.loss_type in ("qr", "wasserstein"):
            batch_size = output.shape[0]
            output = output.view(batch_size, self.output_dim, self.num_quantiles)
        
        return output
        
    def act(self, inputs, role=""):
        """Select actions based on Q-values (SKRL interface)"""
        # Reset noise before action selection for exploration
        if self.training:
            self.reset_noise()
        
        q_values = self.forward(inputs["states"])
        
        if self.loss_type in ("qr", "wasserstein"):
            # For quantile regression, average across quantiles to get Q-values
            q_values = q_values.mean(dim=-1)
        
        # Select action with highest Q-value (noise provides exploration)
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

    def set_noise_scale(self, scale):
        """Dynamically adjust noise scale during training"""
        def _set_noise_scale(module):
            if isinstance(module, NoisyLinear):
                module.weight_sigma.data *= scale
                module.bias_sigma.data *= scale
        
        self.apply(_set_noise_scale)

    def summary(self, input_size=None):
        """Print model summary"""
        if input_size is None:
            input_size = (1, self.input_dim)
        
        print("=== Noisy Q-Network Architecture Summary ===")
        print(f"Loss type: {self.loss_type}")
        print(f"Input dim: {self.input_dim}")
        print(f"Output dim: {self.output_dim}")
        print(f"Use attention: {self.use_attention}")
        print(f"Noise std init: {self.std_init}")
        print(f"Noisy layers only: {self.noisy_layers_only}")
        
        if self.loss_type in ("qr", "wasserstein"):
            print(f"Num quantiles: {self.num_quantiles}")
            print(f"Total output size: {self.output_dim * self.num_quantiles}")
        
        print("\nBackbone:")
        summary(self.backbone, input_size=input_size)
        
        if self.attention is not None:
            print("\nAttention:")
            summary(self.attention, input_size=(input_size[0], self.final_feature_size))
        
        print("\nNoisy Head:")
        summary(self.head, input_size=(input_size[0], self.final_feature_size))
        print("=" * 50)
