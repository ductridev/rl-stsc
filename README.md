# rl-stsc

Reinforcement Learning-Based Smart Traffic Signal Control for Urban Congestion Reduction Model

## Agent Architecture

We are aiming to multi-agent reinforcement learning approach, where each intersection is treated as an independent agent with a unique action space (number of traffic light phase multiply with number of green duration deltas). To handle the problem, we structure the model as follow:

### Backbone

- The backbone is a shared feature extractor for all intersections.

- It processes the state representation of any intersection and encodes it into a high-level latent representation.

#### Backbone architecture

- `Linear(input_dim → 128)`
- `ReLU`
- Repeated `num_layers` times:
  - `Linear(128 → 128)`
  - `ReLU`

This shared backbone enables all intersections to benefit from global learning patterns in traffic dynamics, even though their control spaces differ.

### Attention Layer

- After the backbone, a self-attention mechanism is applied to enhance feature representation.
- By default, a Squeeze-and-Excitation Network (SENet) is used as the attention module, operating on the 256-dimensional feature vector.
- This allows the model to adaptively recalibrate channel-wise feature responses, improving the agent's ability to focus on important traffic features.

### Multiple Heads

- The heads are individual output layers tailored for each unique action space (number of valid phase-duration combinations).
- Each head is a simple `Linear(128 → output_dim)` layer.
- The output dimension is determined by:

  ```python
  output_dims = list(dict.fromkeys([len(phases) for intersection in scenario]))
  ```

  which ensures unique heads for unique action sizes.

- The correct head is selected at runtime based on the current intersection’s configuration.

#### Example

If intersection A has 3 phases → `action_dim = 3`
If intersection B has 5 phases → `action_dim = 5`
Then heads = `{ '3': Linear(128 → 3), '5': Linear(128 → 5) }`

### Forward Pass Logic

```python
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
```

Each agent (intersection) will:

1. Encode its state via the shared backbone.
2. Enhance features via the attention layer.
3. Use its corresponding head (determined by its action space) to predict Q-values.

### DESRA Integration

We integrate **DESRA** (DEcentralized Spillback Resistant Acyclic) as a rule-based policy module that uses shockwave theory to compute optimal traffic phases and green times based on queue lengths and downstream congestion.

#### DQN + DESRA

- DESRA’s selected phase ID and green time are appended to the DQN agent’s state input.
- The DQN model learns to interpret DESRA’s decision as guidance, and can either follow it or override it based on long-term rewards.
- This hybrid architecture offers the stability of rule-based control and the adaptivity of deep reinforcement learning.

#### State Format

The input to the DQN model becomes:

```python
[...per-phase states..., desra_phase_idx]
```
Where per-phase states will be
```python
[waiting_time, num_vehicles, highest_queue_length]
```

- This maintains compatibility with the existing architecture while improving spillback awareness and decision robustness.

### Accident Integration

We integrate the Accident Manager module to simulate traffic accidents at specific junctions or edges during the simulation by random stop
a vehicle inside a junction or edge for a period of time and remove it after the duration is over.

#### Accident Format

```python
[start_step, duration, junction_id_list, edge_id_list]
```

#### Accident Impact

- Accidents introduce new dynamics, such as blocked junctions or edges, increased waiting times, and reduced throughput.
- The agent must explore and learn how to handle both normal traffic conditions and accident scenarios.
- The reward function is updated to penalize high waiting times, queue lengths, and stopped vehicles caused by accidents.
- Training duration may need to be increased to allow the agent to learn effective strategies for handling accidents.
### Benefits

- Parameter sharing improves generalization across intersections.
- Attention mechanism helps the agent focus on the most relevant traffic features.
- Multiple heads ensure flexibility for multiple action spaces.
- The combination between DESRA-DQN enhances stability in congestion and accelerates learning.
- Supports scalable learning as city-size grows (many intersections).
- Allows the system to evaluate the robustness of traffic signal control under unexpected disruptions like accident.


### Training Logic
See [TRAINING.md](TRAINING.md) for details on batch training, target network, and experience replay. After each training run, a snapshot of the configuration used is automatically saved as `config_snapshot.yaml` (YAML format) in the results folder for reproducibility.

### Testing and Evaluation
After each testing run, a snapshot of the configuration is also saved as `config_snapshot.yaml` (YAML format) in the results folder.

### Modularization & SKRL Integration
The codebase is now fully modularized, with SKRL-based agent management, factory patterns for environment/model creation, and clean separation of concerns. See [MODULARIZATION_STATUS.md](MODULARIZATION_STATUS.md) for details.

### Model Saving/Loading
Models are saved per traffic light with the convention: `skrl_model_{traffic_light_id}_episode_{episode}.pt` and config snapshots are saved alongside results.

### Project Methodology Summary
See [SUMMARY.md](SUMMARY.md) for a high-level overview.

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

© 2025 [@trihdde170376](https://github.com/ductridev), [@anhndde170542](https://github.com/Anhsturdy), [@hungnpde170109](https://github.com/NekoTom12343)
