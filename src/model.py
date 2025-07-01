import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from typing import Optional

class DQN(nn.Module):
    def __init__(self, num_layers, batch_size, learning_rate = 0.0001, input_dim = 4, output_dims = [15, 15, 15], gamma = 0.99, device = 'cpu'):
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
        self.device = device

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
        backbone.append(nn.Linear(128, 256))
        backbone.append(nn.ReLU())

        for _ in range(self.num_layers):
            backbone.append(nn.Linear(256, 256))
            backbone.append(nn.ReLU())

        heads = nn.ModuleDict()
        for dim in self._output_dims:
            heads[str(dim)] = nn.Linear(256, dim)

        return backbone, heads
    
    def summary(self):
        """
        Print a summary of the model.
        """
        print("Backbone summary:")
        summary(self.backbone, input_size=(self.batch_size, self.input_dim))

        for dim in self._output_dims:
            print(f"\nHead summary for output_dim={dim}:")
            summary(self.heads[str(dim)], input_size=(self.batch_size, 256))

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

        if not isinstance(x, torch.Tensor):
            x = np.reshape(x, [1, self.input_dim])
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        else:
            x = torch.reshape(x, [1, self.input_dim])
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
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        return self.forward(x, output_dim)
    
    def train_batch(self, states, actions, rewards, next_states, output_dim=15, done=False, target_net:Optional["DQN"]=None):
        """
        Train the model on a batch of experiences.

        Args:
            states (ndarray): Batch of states.
            actions (ndarray): Batch of actions.
            rewards (ndarray): Batch of rewards.
            next_states (ndarray): Batch of next states.
            dones (ndarray): Batch of done flags.
            output_dim (int): Output dimension (action space size).

        Returns:
            dict: Batch training metrics (avg loss, avg q_value, etc.)
        """
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        # Forward pass
        q_values = self.predict_batch(states, output_dim)             # [B, output_dim]
        with torch.no_grad():
            if target_net is not None:
                next_q_values = target_net.predict_batch(next_states, output_dim)
            else:
                next_q_values = self.predict_batch(next_states, output_dim)  # fallback

        # Get Q-value for taken actions: Q(s,a)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-value
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_value = rewards + (1 - done) * self.gamma * max_next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(q_value, target_q_value.detach())

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "avg_q_value": q_value.mean().item(),
            "avg_max_next_q_value": max_next_q_values.mean().item(),
            "avg_target": target_q_value.mean().item()
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