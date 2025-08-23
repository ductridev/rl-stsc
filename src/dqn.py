from skrl.models.torch import Model
import torch.nn as nn

class DQNModel(Model):
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n),
        )

    def forward(self, states, actions=None):
        return self.net(states)