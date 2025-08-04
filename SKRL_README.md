# SKRL-based Traffic Signal Control

This directory contains an adaptation of the original SUMO traffic signal control simulation to use the SKRL (Scikit-Reinforcement Learning) library's DQN implementation instead of the custom DQN in `model.py`.

## Key Files

### Original Implementation
- `src/simulation.py` - Original simulation using custom DQN from `src/model.py`
- `src/model.py` - Custom DQN implementation
- `src/memory.py` - Custom replay memory

### SKRL Implementation
- `src/simulation_skrl.py` - New simulation using SKRL's DQN agent
- `example_skrl_usage.py` - Example showing how to use the SKRL version

## Key Differences

### 1. DQN Agent
- **Original**: Custom DQN implementation with manual network definition and training loops
- **SKRL**: Uses SKRL's optimized DQN agent with built-in experience replay, target network updates, and exploration strategies

### 2. Memory Management
- **Original**: Custom `ReplayMemory` class
- **SKRL**: Uses SKRL's `RandomMemory` with automatic experience storage and sampling

### 3. Training Loop
- **Original**: Manual batch sampling and Q-learning updates
- **SKRL**: Automatic training through agent's `pre_interaction` and `post_interaction` methods

### 4. Model Architecture
- **Original**: Fixed network architecture defined in `model.py`
- **SKRL**: Flexible `DQNNetwork` class that can be easily customized

## Installation

Make sure you have the required dependencies:

```bash
pip install -r requirements.txt
```

The updated `requirements.txt` includes:
- `skrl>=1.2.0` - The SKRL library
- `gymnasium>=0.29.0` - Updated OpenAI Gym interface

## Usage

### Basic Usage

```python
from src.simulation_skrl import SimulationSKRL
from src.visualization import Visualization
from src.accident_manager import AccidentManager

# Initialize components
visualization = Visualization(save_gif=False, path="./results/")
accident_manager = AccidentManager(accident_prob=0.0, accidents=[])

# Create SKRL simulation
simulation = SimulationSKRL(
    visualization=visualization,
    agent_cfg=agent_config,
    max_steps=max_steps,
    traffic_lights=traffic_lights,
    accident_manager=accident_manager,
    # ... other parameters
)

# Run simulation
sim_time, training_time = simulation.run(epsilon=0.1, episode=0)
```

### Using the Example Script

```bash
python example_skrl_usage.py
```

This will load configuration from `config/training_testngatu4.yaml` and run multiple episodes.

## Configuration

The SKRL version uses the same configuration format as the original, with some additional parameters:

```yaml
agent:
  num_layers: 3
  batch_size: 32
  learning_rate: 0.001
  gamma: 0.95
  epsilon: 1.0
  min_epsilon: 0.01
  memory_size_max: 50000
  exploration_timesteps: 50000  # New: exploration schedule
  learning_starts: 1000         # New: when to start learning
  hidden_size: 256             # New: hidden layer size
```

## Model Saving and Loading

The SKRL version provides methods to save and load trained models:

```python
# Save models after training
simulation.save_models(episode=10)

# Load previously trained models
simulation.load_models(episode=10)
```

Models are saved per traffic light with the naming convention:
`{path}skrl_dqn_{traffic_light_id}_episode_{episode}.pt`

## Performance Benefits

The SKRL implementation offers several advantages:

1. **Optimized Training**: SKRL's DQN includes modern optimizations like double DQN, dueling networks, and prioritized experience replay
2. **Memory Efficiency**: Better memory management and automatic cleanup
3. **Stable Training**: Built-in target network updates and exploration scheduling
4. **Extensibility**: Easy to extend with other SKRL algorithms (A3C, PPO, SAC, etc.)

## Comparison with Original

Both implementations should produce comparable results, but the SKRL version may:
- Train more efficiently due to optimized algorithms
- Have more stable convergence
- Provide better exploration strategies
- Offer easier hyperparameter tuning

## Migration Guide

To migrate from the original simulation to SKRL:

1. Replace `from src.simulation import Simulation` with `from src.simulation_skrl import SimulationSKRL`
2. Update your configuration to include SKRL-specific parameters
3. Use the same interface - `simulation.run(epsilon, episode)` works identically
4. Update model saving/loading calls if used

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure SKRL and gymnasium are installed
2. **Memory Issues**: Reduce `memory_size_max` if running out of memory
3. **Training Instability**: Adjust learning rate and exploration parameters

### Performance Tips

1. **Batch Size**: Larger batch sizes (64-128) often work better with SKRL
2. **Learning Rate**: Start with 1e-3 and adjust based on convergence
3. **Memory Size**: 50k-100k samples usually sufficient for traffic control tasks

## Future Extensions

The SKRL framework makes it easy to experiment with:
- Different RL algorithms (A3C, PPO, DDPG)
- Multi-agent setups
- Prioritized experience replay
- Distributional RL
- Meta-learning approaches
