import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):
    def __init__(self, num_layers, batch_size, learning_rate = 0.0001, input_dim = 4, output_dim = 3):
        """
        Initialize the DQN model.

        Args:
            num_layers (int): Number of layers in the neural network.
            batch_size (int): Size of the batches for training.
            learning_rate (float): Learning rate for the optimizer.
            input_dim (int): Dimension of the input state.
            output_dim (int): Dimension of the output action space.
        """
        super(DQN, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = self.__build_model()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def __build_model(self):
        """
        Build the model by defining the layers and their connections.
        """
        self.model = nn.Sequential()
        self.model.append(nn.Linear(self.input_dim, 128))
        self.model.append(nn.ReLU())

        for _ in range(self.num_layers):
            self.model.append(nn.Linear(128, 128))
            self.model.append(nn.ReLU())

        self.model.append(nn.Linear(128, self.output_dim))

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the network.
        """
        x = x.view(self.batch_size, -1)
        x = self.model(x)
        return x
    
    def predict(self, x):
        """
        Predict the Q-values for the given input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Predicted Q-values.
        """
        with torch.no_grad():
            return self.forward(x)
        
    def update(self, state, action, reward, next_state, done):
        """
        Update the model using the given transition.

        Args:
            state (torch.Tensor): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (torch.Tensor): Next state after the action.
            done (bool): Whether the episode has ended.
        """
        # Convert to tensors
        state = torch.FloatTensor(state).unsqueeze(0)
        next_state = torch.FloatTensor(next_state).unsqueeze(0)
        action = torch.LongTensor([action])
        reward = torch.FloatTensor([reward])
        done = torch.FloatTensor([done])

        # Compute Q-values
        q_values = self.forward(state)
        next_q_values = self.predict(next_state)

        # Compute target Q-value
        target_q_value = reward + (1 - done) * 0.99 * torch.max(next_q_values)

        # Compute loss
        loss = F.mse_loss(q_values.gather(1, action.unsqueeze(1)), target_q_value.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
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