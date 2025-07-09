import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
from typing import Optional
from src.SENet_module import SENet
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
            attn_heads (int): Number of attention heads.
        """
        super(DQN, self).__init__()
        self.loss_type = "qr"  
        self.num_layers = num_layers
        self._batch_size = batch_size
        self.learning_rate = learning_rate
        self._input_dim = input_dim
        self._output_dims = list(dict.fromkeys(output_dims))
        self.gamma = gamma
        self.device = device
        self.num_quantiles = 51  
        self.num_atoms     = 51  
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
        for dim in self._output_dims:
            if hasattr(self, "loss_type") and self.loss_type in ("qr", "wasserstein"):
                heads[str(dim)] = nn.Linear(256, dim * self.num_quantiles)
            elif hasattr(self, "loss_type") and self.loss_type == "c51":
                heads[str(dim)] = nn.Linear(256, dim * self.num_atoms)
            else:                                     
                heads[str(dim)] = nn.Linear(256, dim)

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

        assert output_dim is not None, "output_dim must be specified"
        assert isinstance(output_dim, int), "output_dim must be an integer"
        assert output_dim in self._output_dims, f"Invalid output_dim: {output_dim}"

        features = self.backbone(x)  # [B, 256]
        attn_out = self.attn(features)  # Apply SENet, input shape [B, 256]
        head = self.heads[str(output_dim)]
        out = head(attn_out)
        if self.loss_type in ("qr", "wasserstein"):
            return out.view(-1, output_dim, self.num_quantiles)   
        elif self.loss_type == "c51":
            return out.view(-1, output_dim, self.num_atoms)
        else:
            return out  
    
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
            output_dim (int): Output dimension (action space size).

        Returns:
            dict: Batch training metrics (avg loss, avg q_value, etc.)
        """
        # Convert to tensors
        loss_type = self.loss_type 
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)
        online  = self
        # Forward pass
        # Compute target Q-value using DDQN logic
        next_q_values = (
            target_net.predict_batch(next_states, output_dim)
            if target_net is not None else self.predict_batch(next_states, output_dim)
        )
        q_values = self.predict_batch(states, output_dim)

        if loss_type in ("mse", "huber", "weighted"):
            next_actions = torch.argmax(self.predict_batch(next_states, output_dim), dim=1)
            target_q_value = rewards + (1 - done) * self.gamma * next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
         # === Loss computation ===
            if loss_type == "mse":
                loss = F.mse_loss(q_value, target_q_value.detach())

            elif loss_type == "huber":
                loss = F.smooth_l1_loss(q_value, target_q_value.detach(), beta=1.0)

            elif loss_type == "weighted":
                delta = 0.85
                weights = torch.where(q_value < target_q_value,
                                    torch.ones_like(q_value),
                                    torch.clamp(target_q_value / (q_value + 1e-6), min=delta))
                loss = (weights * (q_value - target_q_value.detach()) ** 2).mean()
        # ----------  Quantile Regression (QR-DQN) ----------
        elif loss_type == "qr":
            K = self.num_quantiles
            
            q_dist      = q_values.view(-1, output_dim, K)
            next_q_dist = next_q_values.view(-1, output_dim, K)

            next_a = torch.argmax(next_q_dist.mean(2), dim=1)
            tgt_dist  = rewards.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * \
                        next_q_dist[torch.arange(next_q_dist.size(0)), next_a]   
            pred_dist = q_dist[torch.arange(q_dist.size(0)), actions]            

            u = tgt_dist.unsqueeze(2) - pred_dist.unsqueeze(1)                   
            huber = F.smooth_l1_loss(pred_dist.unsqueeze(1).expand_as(u),
                                    tgt_dist.unsqueeze(2).expand_as(u),
                                    beta=1.0, reduction='none')
            taus = (torch.arange(K, device=self.device).float() + 0.5) / K
            weight = torch.abs(taus.unsqueeze(0) - (u.detach() < 0).float())
            loss = (weight * huber).mean()
            target_q_value = tgt_dist.mean(1)
            q_value = pred_dist.mean(1)   

        # ----------  Categorical (C51) ----------
        # elif loss_type == "c51":
        #     N = self.num_atoms; vmin, vmax = self.v_min, self.v_max
        #     Δz = (vmax - vmin) / (N - 1)
        #     z  = torch.linspace(vmin, vmax, N, device=self.device)
        #     logits      = q_values.view(-1, output_dim, N)
        #     next_logits = next_q_values.view(-1, output_dim, N)
        #     prob        = F.softmax(logits,      dim=2)
        #     prob_next   = F.softmax(next_logits, dim=2)

        #     with torch.no_grad():
        #         next_a = torch.argmax((prob_next * z).sum(2), 1)
        #         p_next_a = prob_next[torch.arange(prob_next.size(0)), next_a]         
        #         Tz = rewards.unsqueeze(1) + (1 - done.unsqueeze(1)) * self.gamma * z
        #         Tz = Tz.clamp(vmin, vmax)
        #         b  = (Tz - vmin) / Δz
        #         l  = b.floor().long(); u = b.ceil().long()
        #         m = torch.zeros_like(p_next_a)
        #         for j in range(N):
        #             l_mask = (l == j); u_mask = (u == j)
        #             m[..., j] += (p_next_a * (u.float() - b))[l_mask]
        #             m[..., j] += (p_next_a * (b - l.float()))[u_mask]

        #     loss = -(m * prob.log()).sum(2).mean()
        #     target_q_value = (m * z).sum(1)    # ✅ Estimate target_q_value
        #     q_value = (prob * z).sum(2).gather(1, actions.unsqueeze(1)).squeeze(1)  # ✅ Add this line

        else:
            raise ValueError(f"Unknown loss_type '{loss_type}'")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
         
        max_next_q_value = next_q_values.max(dim=1)[0]

        return {
            "loss": loss.item(),
            "avg_q_value": q_value.mean().item(),
            "avg_max_next_q_value": max_next_q_value.mean().item(),
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