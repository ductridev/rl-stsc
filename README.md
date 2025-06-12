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

### Multiple Heads

- The heads are individual output layers tailored for each unique action space (number of valid phase-duration combinations).
- Each head is a simple `Linear(128 → output_dim)` layer.
- The output dimension is determined by:
    ```python
    output_dims = list(dict.fromkeys([len(phases) * len(time_deltas) for intersection in scenario]))
    ```

    which ensures unique heads for unique action sizes.
- The correct head is selected at runtime based on the current intersection’s configuration.

#### Example

If intersection A has 3 phases and 5 time deltas → `action_dim = 15`
If intersection B has 5 phases and 5 time deltas → `action_dim = 25`
Then heads = `{ '15': Linear(128 → 15), '25': Linear(128 → 25) }`

### Forward Pass Logic

```python
def forward(self, x, action_dim):
    x = self.backbone(x)
    head = self.heads[str(action_dim)]
    return head(x)
```

Each agent (intersection) will:
1. Encode its state via the shared backbone.
2. Use its corresponding head (determined by its action space) to predict Q-values.

### DESRA Integration

We integrate **DESRA** (DEcentralized Spillback Resistant Acyclic) as a rule-based policy module that uses shockwave theory to compute optimal traffic phases and green times based on queue lengths and downstream congestion.

#### DQN + DESRA
- DESRA’s selected phase ID and green time are appended to the DQN agent’s state input.
- The DQN model learns to interpret DESRA’s decision as guidance, and can either follow it or override it based on long-term rewards.
- This hybrid architecture offers the stability of rule-based control and the adaptivity of deep reinforcement learning.

#### State Format
The input to the DQN model becomes:
``` python
[min_free_capacity, density, waiting_time, queue_length, desra_phase_id, desra_green_time]

```

- This maintains compatibility with the existing architecture while improving spillback awareness and decision robustness.

### Benefits

- Parameter sharing improves generalization across intersections.
- Multiple heads ensure flexibility for multiple action spaces.
- The combination between DESRA-DQN enhances stability in congestion and accelerates learning.
- Supports scalable learning as city-size grows (many intersections).

### Training Logic
Check [TRAINING.md](TRAINING.md)

## License

This project is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/) license.

© 2025 [@trihdde170376](https://github.com/ductridev), [@anhndde170542](https://github.com/Anhsturdy), [@hungnpde170109](https://github.com/NekoTom12343)
