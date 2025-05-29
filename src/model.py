import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary

class DQN(nn.Module):
    def __init__(self, num_layers, batch_size, learning_rate = 0.0001, input_dim = 4, output_dims = [15, 15, 15], gamma = 0.99):
        """
        Initialize the DQN model.

        Args:
            num_layers (int): Number of layers in the neural network.
            batch_size (int): Size of the batches for training.
            learning_rate (float): Learning rate for the optimizer.
            input_dim (int): Dimension of the input state.
            output_dims (list[int]): A list defining multiple output actions space.
        """
        super(DQN, self).__init__()
        self.num_layers = num_layers
        self._batch_size = batch_size
        self.learning_rate = learning_rate
        self._input_dim = input_dim
        self._output_dims = list(dict.fromkeys(output_dims))
        self.gamma = gamma

        # Build model parts: backbone + heads
        self.backbone, self.heads = self.__build_model()

        self.summary()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __build_model(self):
        """
        Build the model parts by defining the layers and their connections.
        We must split model into backbone and heads because heads(a.k.a outputs) are different for each intersection or traffic light
        """
        backbone = nn.Sequential()
        backbone.append(nn.Linear(self.input_dim, 128))
        backbone.append(nn.ReLU())

        for _ in range(self.num_layers):
            backbone.append(nn.Linear(128, 128))
            backbone.append(nn.ReLU())

        heads = nn.ModuleDict()
        for dim in self._output_dims:
            heads[str(dim)] = nn.Linear(128, dim)

        return backbone, heads
    
    def summary(self):
        """
        Print a summary of the model.
        """
        print("Backbone summary:")
        summary(self.backbone, input_size=(self.batch_size, self.input_dim))

        for dim in self._output_dims:
            print(f"\nHead summary for output_dim={dim}:")
            summary(self.heads[str(dim)], input_size=(dim, 128))

    def forward(self, x, output_dim = 15):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.
            output_dim (int): Dimension of the output.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        assert output_dim is not None, "output_dim must be specified"
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert output_dim in self._output_dims, f"Invalid output_dim: {output_dim}"

        features = self.backbone(x)
        head = self.heads[str(output_dim)]
        return head(features)
    
    def predict_one(self, x, output_dim = 15):
        """
        Predict the Q-values for the given input.

        Args:
            x (torch.Tensor): Input tensor.
            output_dim (int): Dimension of the output.

        Returns:
            torch.Tensor: Predicted Q-values.
        """
        assert output_dim is not None, "output_dim must be specified"
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert output_dim in self._output_dims, f"Invalid output_dim: {output_dim}"

        x = np.reshape(x, [1, self.input_dim])
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.forward(x, output_dim)
    
    def predict_batch(self, x, output_dim = 15):
        """
        Predict the Q-values for the given batch of inputs.

        Args:
            x (torch.Tensor): Batch of input tensors.
            output_dim (int): Dimension of the output.

        Returns:
            torch.Tensor: Predicted Q-values.
        """
        assert output_dim is not None, "output_dim must be specified"
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert output_dim in self._output_dims, f"Invalid output_dim: {output_dim}"

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return self.forward(x, output_dim)

    def update(self, state, action, reward, next_state, done = False, output_dim = 15):
        """
        Update the model using the given transition.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state after the action.
            done (bool): Whether the episode has ended.
        Returns:
            dict: Contains loss, q_value, max_next_q_value, and target.
        """
        assert output_dim is not None, "output_dim must be specified"
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert output_dim in self._output_dims, f"Invalid output_dim: {output_dim}"

        # Convert to tensors
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        # Compute Q-values
        q_values = self.forward(state, output_dim)
        next_q_values = self.predict_one(next_state, output_dim)

        q_value = q_values[action]
        max_next_q_value = torch.max(next_q_values)

        # Compute target Q-value
        target_q_value = reward + (1 - done) * self.gamma * torch.max(next_q_values)

        # Compute loss
        loss = F.mse_loss(q_value, target_q_value)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "q_value": q_value.item(),
            "max_next_q_value": max_next_q_value.item(),
            "target": target_q_value.item(),
        }
    
    def save(self, path):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    @property
    def input_dim(self):
        """
        Get the input dimension of the model.
        """
        return self._input_dim
    
    @property
    def output_dim(self):
        """
        Get the output dimension of the model.
        """
        return self._output_dim
    
    @property
    def batch_size(self):
        """
        Get the batch size of the model.
        """
        return self._batch_size