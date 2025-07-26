import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from typing import Optional
from src.SENet_module import SENet


class DQN(nn.Module):
    def __init__(
        self,
        num_layers,
        batch_size,
        learning_rate=0.0001,
        input_dim=4,
        output_dims=[15, 15, 15],
        gamma=0.99,
        loss_type="qr",
        device="cpu",
        max_phases=10,
    ):
        """
        Initialize the DQN model.

        Args:
            num_layers (int): Number of layers in the neural network.
            batch_size (int): Size of the batches for training.
            learning_rate (float): Learning rate for the optimizer.
            input_dim (int): Dimension of the input state.
            output_dims (list[int]): A list defining multiple output actions space.
            attn_heads (int): Number of attention heads.
        """
        super(DQN, self).__init__()
        self.loss_type = loss_type
        self.num_layers = num_layers
        self._batch_size = batch_size
        self.learning_rate = learning_rate
        self._input_dim = max_phases * input_dim + 2
        self._output_dims = list(dict.fromkeys(output_dims))
        self.gamma = gamma
        self.device = device
        self.num_quantiles = 100
        self.v_min, self.v_max = -10.0, 10.0
        self.backbone, self.attn, self.heads = self.__build_model()

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

        attn = SENet(channel=256)  # Replace multihead attention

        heads = nn.ModuleDict()
        green_heads = nn.ModuleDict()

        for dim in self._output_dims:
            if hasattr(self, "loss_type") and self.loss_type in ("qr", "wasserstein"):
                heads[str(dim)] = nn.Linear(256, dim * self.num_quantiles)
            elif hasattr(self, "loss_type") and self.loss_type == "c51":
                heads[str(dim)] = nn.Linear(256, dim * self.num_atoms)
            else:
                heads[str(dim)] = nn.Linear(256, dim)
            green_heads[str(dim)] = nn.Linear(256, dim)

        self.green_heads = green_heads
        return backbone, attn, heads

    def summary(self):
        """
        Print a summary of the model.
        """
        print("Backbone summary:")
        summary(self.backbone, input_size=(self.batch_size, self.input_dim))
        print("\nAttention layer: SENet(channel=256)")
        summary(self.attn, input_size=(self.batch_size, 256))
        for dim in self._output_dims:
            print(f"\nHead summary for output_dim={dim}:")
            summary(self.heads[str(dim)], input_size=(self.batch_size, 256))

    def forward(self, x, output_dim=15):
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

        features = self.backbone(x)  # [B, 256]
        attn_out = self.attn(features)  # Apply SENet, input shape [B, 256]

        q_head = self.heads[str(output_dim)]
        green_head = self.green_heads[str(output_dim)]

        q_out = q_head(attn_out)  # Q-values or distributions
        green_out = green_head(attn_out)  # Green time predictions

        if self.loss_type in ("qr", "wasserstein"):
            q_out = q_out.view(-1, output_dim, self.num_quantiles)

        elif self.loss_type == "c51":
            q_out = q_out.view(-1, output_dim, self.num_atoms)

        return q_out, green_out

    def predict_one(self, x, output_dim=15):
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

    def predict_batch(self, x, output_dim=15):
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

    def train_batch(
        self,
        states,
        actions,
        rewards,
        next_states,
        green_targets,
        output_dim=15,
        done=False,
        target_net: Optional["DQN"] = None,
        alpha: float = 0.8,
    ):
        """
        Train the model on a batch of experiences.

        Args:
            states (ndarray): Batch of states.
            actions (ndarray): Batch of actions (phase indices).
            rewards (ndarray): Batch of rewards.
            next_states (ndarray): Batch of next states.
            green_targets (ndarray): Ground truth green times (used or DESRA).
            output_dim (int): Output dimension (number of valid phases).
            done (bool or np.ndarray): Whether episode terminated.
            target_net (DQN): Optional target network for DDQN.
            alpha (float): Alpha value for soft target network update.

        Returns:
            dict: Batch training metrics (avg loss, avg q_value, etc.)
        """
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(
            self.device
        )
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        green_targets = torch.tensor(np.array(green_targets), dtype=torch.float32).to(
            self.device
        )
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        loss_type = self.loss_type
        online = self

        # Forward pass
        # Compute target Q-value using DDQN logic
        q_values, green_preds = self.predict_batch(
            states, output_dim
        )  # [B, A, K] and [B, A]
        next_q_values, _ = (
            target_net.predict_batch(next_states, output_dim)
            if target_net is not None
            else self.predict_batch(next_states, output_dim)
        )

        if loss_type in ("mse", "huber", "weighted"):
            q_value_all = q_values  # [B, A]
            next_q_all = next_q_values  # [B, A]

            next_actions = torch.argmax(
                self.predict_batch(next_states, output_dim)[0], dim=1
            )
            target_q_value = rewards + (1 - done) * self.gamma * next_q_all.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            q_value = q_value_all.gather(1, actions.unsqueeze(1)).squeeze(1)

            # === Loss computation ===
            if loss_type == "mse":
                q_loss = F.mse_loss(q_value, target_q_value.detach())

            elif loss_type == "huber":
                q_loss = F.smooth_l1_loss(q_value, target_q_value.detach(), beta=1.0)

            elif loss_type == "weighted":
                delta = 0.85
                weights = torch.where(
                    q_value < target_q_value,
                    torch.ones_like(q_value),
                    torch.clamp(target_q_value / (q_value + 1e-6), min=delta),
                )
                q_loss = (weights * (q_value - target_q_value.detach()) ** 2).mean()
        elif loss_type == "qr":
            K = self.num_quantiles

            q_dist = q_values.view(
                -1, output_dim, K
            )  # [batch, num_actions, num_quantiles]
            next_q_dist = next_q_values.view(-1, output_dim, K)

            # Get the best next action by averaging quantiles, then taking argmax
            next_actions = torch.argmax(next_q_dist.mean(2), dim=1)

            # Compute target quantile values: rewards + gamma * next_q_values
            tgt_dist = (
                rewards.unsqueeze(1)
                + (1 - done.unsqueeze(1))
                * self.gamma
                * next_q_dist[torch.arange(next_q_dist.size(0)), next_actions]
            )

            # Get predicted quantiles for actions actually taken
            pred_dist = q_dist[torch.arange(q_dist.size(0)), actions]

            # Quantile Huber loss
            # Compute pairwise differences between target and predicted quantiles
            u = tgt_dist.unsqueeze(2) - pred_dist.unsqueeze(1)  # [B, K, K]

            huber = F.smooth_l1_loss(
                pred_dist.unsqueeze(1).expand_as(u),
                tgt_dist.unsqueeze(2).expand_as(u),
                beta=1.0,
                reduction="none",
            )

            # Compute quantile fractions taus = (0.5 + i) / K for i in 0..K-1
            taus = (torch.arange(K, device=self.device).float() + 0.5) / K

            # Quantile weight: |τ - 𝟙{u < 0}|
            # Encourages over-estimation and under-estimation sensitivity
            weight = torch.abs(taus.unsqueeze(0) - (u.detach() < 0).float())
            # Final loss: average over batch, quantiles
            q_loss = (weight * huber).mean()

            # Mean Q-values for logging
            target_q_value = tgt_dist.mean(1)
            q_value = pred_dist.mean(1)

        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'")

        # === Green Time Prediction Loss ===
        green_preds_taken = green_preds[
            torch.arange(green_preds.size(0)), actions
        ]  # [B]

        # Clamp predictions and targets for stability
        green_preds_taken = green_preds_taken.clamp(min=5.0, max=60.0)
        green_targets = green_targets.clamp(min=5.0, max=60.0)

        green_loss = F.mse_loss(green_preds_taken, green_targets)

        # === Total Loss ===
        total_loss = q_loss + alpha * green_loss

        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        max_next_q_value = next_q_values.max(dim=1)[0]

        return {
            "q_loss": q_loss.item(),
            "green_loss": green_loss.item(),
            "total_loss": total_loss.item(),
            "avg_q_value": q_value.mean().item(),
            "avg_max_next_q_value": max_next_q_value.mean().item(),
            "avg_target": target_q_value.detach().mean().item(),
        }

    def save(self, path):
        """
        Save the model to the specified path.

        Args:
            path (str): Path to save the model.
        """
        torch.save(self.state_dict(), path)

    def save_checkpoint(self, path, episode=None, epsilon=None):
        """
        Save the complete model checkpoint including optimizer state for continuing training.

        Args:
            path (str): Path to save the checkpoint.
            episode (int, optional): Current episode number.
            epsilon (float, optional): Current epsilon value.
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "episode": episode,
            "epsilon": epsilon,
            "model_config": {
                "num_layers": self.num_layers,
                "batch_size": self._batch_size,
                "learning_rate": self.learning_rate,
                "input_dim": self._input_dim,
                "output_dims": self._output_dims,
                "gamma": self.gamma,
                "loss_type": self.loss_type,
                "num_quantiles": self.num_quantiles,
                "num_atoms": self.num_atoms,
                "v_min": self.v_min,
                "v_max": self.v_max,
            },
        }
        torch.save(checkpoint, path)

    def load(self, path, for_training=False):
        """
        Load the model from the specified path.

        Args:
            path (str): Path to load the model from.
            for_training (bool): If True, keep model in training mode. If False, set to eval mode.
        """
        self.load_state_dict(torch.load(path, map_location=self.device))
        if for_training:
            self.train()  # Set to training mode for continued training
        else:
            self.eval()  # Set to evaluation mode for inference

    def load_checkpoint(self, path):
        """
        Load the complete model checkpoint including optimizer state for continuing training.

        Args:
            path (str): Path to load the checkpoint from.

        Returns:
            dict: Dictionary containing episode and epsilon if available.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train()  # Set to training mode

        # Return training metadata
        return {
            "episode": checkpoint.get("episode", 0),
            "epsilon": checkpoint.get("epsilon", 1.0),
        }

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
