"""
Q-Network model for DQN using SKRL with advanced features from the original model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from skrl.models.torch import Model, DeterministicMixin
from torchinfo import summary
from typing import Any, Mapping, Optional, Tuple, Union
from src.SENet_module import SENet, SENetFC

class MLP(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 32),
                                 nn.ReLU(),
                                 nn.Linear(32, self.num_actions))

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}
    
    def summary(self, input_size=None):
        """Print model summary"""
        return
        if input_size is None:
            input_size = (1, self.input_dim)
        
        print("=== Q-Network Architecture Summary ===")
        print(f"Input dim: {self.input_dim}")
        print(f"Output dim: {self.output_dim}")
        print(f"Use attention: {self.use_attention}")
        
        print("\nBackbone:")
        summary(self.backbone, input_size=input_size)
        
        if self.attention is not None:
            print("\nAttention:")
            summary(self.attention, input_size=(input_size[0], self.final_feature_size))
        
        print("\nHead:")
        summary(self.head, input_size=(input_size[0], self.final_feature_size))
        print("=" * 40)

class QNetwork(Model):
    """Advanced Q-network model for DQN using SKRL with features from original model"""

    def __init__(self, observation_space, action_space, device="cpu", **kwargs):
        Model.__init__(self, observation_space, action_space, device)

        # Basic configuration
        self.input_dim = kwargs.get("input_dim", self.num_observations)
        self.output_dim = kwargs.get("output_dim", self.num_actions)
        self.device = device
        
        # Architecture configuration
        self.num_layers = kwargs.get("num_layers", 3)
        self.hidden_size = kwargs.get("hidden_size", 256)
        self.use_attention = kwargs.get("use_attention", True)

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
            layers.append(nn.Linear(self.input_dim, self.hidden_size))
            if self.batch_norm:
                layers.append(nn.LayerNorm(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            # Second layer
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
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
        """Build attention mechanism using improved SENet"""
        if self.use_attention:
            # Use the advanced SENet for fully connected features
            return SENet(
                channel=self.final_feature_size,
                reduction=2,
                force_1d=True,
            )
        return None

    def _build_head(self):
        """Build the output head based on loss type"""
        # Standard Q-values: one output per action
        head = nn.Linear(self.final_feature_size, self.output_dim)
        
        return head

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
        
        return output
    
    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act deterministically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is ``None``. The third component is a dictionary containing extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, _, outputs = model.act({"states": states})
            >>> print(actions.shape, outputs)
            torch.Size([4096, 1]) {}
        """
        # map from observations/states to actions
        actions, outputs = self.compute(inputs, role)

        return actions, None, outputs

    def summary(self, input_size=None):
        """Print model summary"""
        if input_size is None:
            input_size = (1, self.input_dim)
        
        print("=== Q-Network Architecture Summary ===")
        print(f"Input dim: {self.input_dim}")
        print(f"Output dim: {self.output_dim}")
        print(f"Use attention: {self.use_attention}")
        
        print("\nBackbone:")
        summary(self.backbone, input_size=input_size)
        
        if self.attention is not None:
            print("\nAttention:")
            summary(self.attention, input_size=(input_size[0], self.final_feature_size))
        
        print("\nHead:")
        summary(self.head, input_size=(input_size[0], self.final_feature_size))
        print("=" * 40)

class TestQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, model_path, device="cpu",
                 num_layers=3, hidden_size=256, use_attention=True,
                 dropout_rate=0.0, batch_norm=False, layer_sizes=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.layer_sizes = layer_sizes

        # Build backbone, attention, head (same as QNetwork)
        self.backbone = self._build_backbone().to(self.device)
        self.attention = self._build_attention().to(self.device) if self.use_attention else None
        self.head = self._build_head().to(self.device)

        # Load weights
        self._load_my_model(model_path)

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
            layers.append(nn.Linear(self.input_dim, self.hidden_size))
            if self.batch_norm:
                layers.append(nn.LayerNorm(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            
            # Second layer
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
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
        """Build attention mechanism using improved SENet"""
        if self.use_attention:
            # Use the advanced SENet for fully connected features
            return SENet(
                channel=self.final_feature_size,
                reduction=2,
                force_1d=True,
            )
        return None

    def _build_head(self):
        """Build the output head based on loss type"""
        # Standard Q-values: one output per action
        head = nn.Linear(self.final_feature_size, self.output_dim)
        
        return head

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)  # [B, feature_size]
        
        # Apply attention if enabled
        if self.attention is not None:
            features = self.attention(features)
        
        # Output head
        output = self.head(features)
        
        return output

    def _load_my_model(self, model_file_path):
        if os.path.isfile(model_file_path):
            checkpoint = torch.load(model_file_path, map_location=self.device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.load_state_dict(checkpoint["model_state_dict"])
            elif isinstance(checkpoint, dict):
                self.load_state_dict(checkpoint)
            else:
                # whole model was saved
                self.load_state_dict(checkpoint.state_dict())
            self.eval()
        else:
            sys.exit("❌ Model file not found: " + model_file_path)

    def predict_one(self, state):
        """Predict Q-values for a single input (compatibility method)"""
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.tensor(np.reshape(state, [1, self.input_dim]), dtype=torch.float32).to(self.device)
            else:
                state = torch.reshape(state, [1, self.input_dim]).to(self.device)

            q_values = self.forward(state)
        return q_values.detach().cpu().numpy().squeeze()