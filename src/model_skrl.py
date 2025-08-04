"""
SKRL-compatible neural network models for traffic signal control.
This module provides neural network architectures optimized for DQN and other RL algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from skrl.models.torch import Model, DeterministicMixin, CategoricalMixin


class DQNNetwork(DeterministicMixin, Model):
    """
    Deep Q-Network for traffic signal control.
    
    This network takes traffic state observations and outputs Q-values for each
    possible traffic light phase action.
    """
    
    def __init__(self, observation_space, action_space, device, **kwargs):
        """
        Initialize the DQN network.
        
        Args:
            observation_space: Observation space from the environment
            action_space: Action space from the environment  
            device: PyTorch device (cpu/cuda)
            **kwargs: Additional configuration parameters
        """
        # Extract custom parameters before calling parent init
        self.num_layers = kwargs.pop("num_layers", 3)
        self.hidden_size = kwargs.pop("hidden_size", 256)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.1)
        self.use_batch_norm = kwargs.pop("use_batch_norm", False)
        self.use_dueling = kwargs.pop("use_dueling", False)
        
        # Call parent init with remaining kwargs
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)

        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n

        # Build the network
        self.net = self._build_network()
        
        # Initialize weights
        self._initialize_weights()

    def _build_network(self) -> nn.Module:
        """Build the neural network architecture."""
        layers = []
        
        # Input layer
        input_size = self.input_dim
        
        # First hidden layer
        layers.append(nn.Linear(input_size, 128))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(128))
        layers.append(nn.ReLU())
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))
        
        # Second hidden layer
        layers.append(nn.Linear(128, self.hidden_size))
        if self.use_batch_norm:
            layers.append(nn.BatchNorm1d(self.hidden_size))
        layers.append(nn.ReLU())
        if self.dropout_rate > 0:
            layers.append(nn.Dropout(self.dropout_rate))

        # Additional hidden layers
        for _ in range(self.num_layers - 2):  # -2 because we already added 2 layers
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            if self.use_batch_norm:
                layers.append(nn.BatchNorm1d(self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))

        # Output layer (no activation for Q-values)
        layers.append(nn.Linear(self.hidden_size, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Q-values for given inputs.
        
        Args:
            inputs: Dictionary with "states" key containing state tensor
            role: Role of the model (not used in DQN)
        
        Returns:
            Tuple of (q_values_tensor, empty_dict)
        """
        states = inputs["states"]
        q_values = self.net(states)
        return q_values, {}

    def random_act(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, None, Dict[str, Any]]:
        """
        Select random actions for exploration.
        
        Args:
            inputs: Dictionary with "states" key containing state tensor
            role: Role of the model
            
        Returns:
            Tuple of (actions, None, empty_dict)
        """
        # Handle case where inputs["states"] might be a dict itself
        states = inputs["states"]
        if isinstance(states, dict):
            states = states["states"]  # Extract the actual tensor
        
        batch_size = states.shape[0]
        actions = torch.randint(0, self.output_dim, (batch_size, 1), device=self.device)
        return actions, None, {}


class DuelingDQNNetwork(DeterministicMixin, Model):
    """
    Dueling Deep Q-Network for improved value estimation.
    
    Separates state value and advantage estimation for better learning.
    """
    
    def __init__(self, observation_space, action_space, device, **kwargs):
        """Initialize the Dueling DQN network."""
        # Extract custom parameters before calling parent init
        self.num_layers = kwargs.pop("num_layers", 3)
        self.hidden_size = kwargs.pop("hidden_size", 256)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.1)
        self.use_batch_norm = kwargs.pop("use_batch_norm", False)
        self.use_dueling = kwargs.pop("use_dueling", True)  # This is a dueling network
        
        # Call parent init with remaining kwargs
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)
        
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n

        # Shared feature layers
        self.feature_layers = self._build_feature_layers()
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.output_dim)
        )
        
        self._initialize_weights()

    def _build_feature_layers(self) -> nn.Module:
        """Build shared feature extraction layers."""
        layers = []
        input_size = self.input_dim
        
        # Feature extraction layers
        layers.append(nn.Linear(input_size, 128))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(128, self.hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in [self.feature_layers, self.value_stream, self.advantage_stream]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute Q-values using dueling architecture.
        
        Args:
            inputs: Dictionary with "states" key
            role: Role of the model
            
        Returns:
            Tuple of (q_values, empty_dict)
        """
        states = inputs["states"]
        
        # Shared features
        features = self.feature_layers(states)
        
        # Value and advantage streams
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values, {}

    def random_act(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, None, Dict[str, Any]]:
        """
        Select random actions for exploration.
        
        Args:
            inputs: Dictionary with "states" key containing state tensor
            role: Role of the model
            
        Returns:
            Tuple of (actions, None, empty_dict)
        """
        # Handle case where inputs["states"] might be a dict itself
        states = inputs["states"]
        if isinstance(states, dict):
            states = states["states"]  # Extract the actual tensor
        
        batch_size = states.shape[0]
        actions = torch.randint(0, self.output_dim, (batch_size, 1), device=self.device)
        return actions, None, {}


class PolicyNetwork(CategoricalMixin, Model):
    """
    Policy network for actor-critic methods (A3C, PPO, etc.).
    
    Outputs action probabilities for discrete action spaces.
    """
    
    def __init__(self, observation_space, action_space, device, **kwargs):
        """Initialize the policy network."""
        # Extract custom parameters before calling parent init
        self.num_layers = kwargs.pop("num_layers", 3)
        self.hidden_size = kwargs.pop("hidden_size", 256)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.1)
        self.use_batch_norm = kwargs.pop("use_batch_norm", False)
        self.use_dueling = kwargs.pop("use_dueling", False)
        
        # Call parent init with remaining kwargs
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        CategoricalMixin.__init__(self, unnormalized_log_prob=True)
        
        self.input_dim = observation_space.shape[0]
        self.output_dim = action_space.n

        self.net = self._build_network()
        self._initialize_weights()

    def _build_network(self) -> nn.Module:
        """Build the policy network."""
        layers = []
        input_size = self.input_dim
        
        layers.append(nn.Linear(input_size, 128))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(128, self.hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        
        # Output layer with log-softmax for categorical distribution
        layers.append(nn.Linear(self.hidden_size, self.output_dim))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute action log-probabilities.
        
        Args:
            inputs: Dictionary with "states" key
            role: Role of the model
            
        Returns:
            Tuple of (log_probabilities, empty_dict)
        """
        states = inputs["states"]
        logits = self.net(states)
        return logits, {}


class ValueNetwork(DeterministicMixin, Model):
    """
    Value network for actor-critic methods.
    
    Estimates state values V(s).
    """
    
    def __init__(self, observation_space, action_space, device, **kwargs):
        """Initialize the value network."""
        # Extract custom parameters before calling parent init
        self.num_layers = kwargs.pop("num_layers", 3)
        self.hidden_size = kwargs.pop("hidden_size", 256)
        self.dropout_rate = kwargs.pop("dropout_rate", 0.1)
        self.use_batch_norm = kwargs.pop("use_batch_norm", False)
        self.use_dueling = kwargs.pop("use_dueling", False)
        
        # Call parent init with remaining kwargs
        Model.__init__(self, observation_space, action_space, device, **kwargs)
        DeterministicMixin.__init__(self)
        
        self.input_dim = observation_space.shape[0]

        self.net = self._build_network()
        self._initialize_weights()

    def _build_network(self) -> nn.Module:
        """Build the value network."""
        layers = []
        input_size = self.input_dim
        
        layers.append(nn.Linear(input_size, 128))
        layers.append(nn.ReLU())
        
        layers.append(nn.Linear(128, self.hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(self.num_layers - 2):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
        
        # Output single value
        layers.append(nn.Linear(self.hidden_size, 1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def compute(self, inputs: Dict[str, torch.Tensor], role: str = "") -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute state values.
        
        Args:
            inputs: Dictionary with "states" key
            role: Role of the model
            
        Returns:
            Tuple of (values, empty_dict)
        """
        states = inputs["states"]
        values = self.net(states)
        return values, {}


class ModelFactory:
    """
    Factory class for creating neural network models.
    """
    
    @staticmethod
    def create_dqn_models(observation_space, action_space, device, **kwargs) -> Dict[str, Model]:
        """
        Create DQN models (policy and target).
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            device: PyTorch device
            **kwargs: Additional model configuration
            
        Returns:
            Dictionary with 'policy' and 'target_policy' models
        """
        use_dueling = kwargs.get("use_dueling", False)
        
        if use_dueling:
            policy_model = DuelingDQNNetwork(observation_space, action_space, device, **kwargs)
            target_model = DuelingDQNNetwork(observation_space, action_space, device, **kwargs)
        else:
            policy_model = DQNNetwork(observation_space, action_space, device, **kwargs)
            target_model = DQNNetwork(observation_space, action_space, device, **kwargs)
        
        # Initialize target network with same weights as policy
        target_model.load_state_dict(policy_model.state_dict())
        
        return {
            "q_network": policy_model,
            "target_q_network": target_model
        }
    
    @staticmethod
    def create_actor_critic_models(observation_space, action_space, device, **kwargs) -> Dict[str, Model]:
        """
        Create actor-critic models for PPO, A3C, etc.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space
            device: PyTorch device
            **kwargs: Additional model configuration
            
        Returns:
            Dictionary with 'policy' and 'value' models
        """
        shared_network = kwargs.get("shared_network", False)
        
        if shared_network:
            # TODO: Implement shared network architecture
            raise NotImplementedError("Shared network not yet implemented")
        else:
            policy_model = PolicyNetwork(observation_space, action_space, device, **kwargs)
            value_model = ValueNetwork(observation_space, action_space, device, **kwargs)
        
        return {
            "policy": policy_model,
            "value": value_model
        }
