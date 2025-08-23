# RL-STSC: Reinforcement Learning for Smart Traffic Signal Control

## Project Objectives and RL Challenges

The primary objective of RL-STSC is to develop an intelligent traffic signal control system that optimizes traffic flow and reduces congestion in urban environments. The project leverages reinforcement learning to enable adaptive, data-driven decision-making for traffic lights at complex intersections. 

**Key RL Challenges addressed:**
- **Uncertainty and Variability:** Traffic patterns are highly dynamic and unpredictable. The RL agent must learn to make robust decisions in the face of fluctuating vehicle flows, unexpected events (e.g., accidents, pedestrian crossings), and incomplete information.
- **Delayed and Sparse Rewards:** The impact of a traffic signal decision may not be immediately observable, requiring the agent to learn from delayed feedback.
- **Multi-Agent and Multi-Intersection Coordination:** Managing multiple intersections simultaneously introduces additional complexity, as actions at one intersection can affect traffic at others.
- **Exploration vs. Exploitation:** The agent must balance trying new strategies (exploration) with leveraging learned policies (exploitation) to optimize long-term performance.


## Experiment Tracking

After each training or testing run, a snapshot of the configuration is automatically saved as `config_snapshot.yaml` (YAML format) in the results folder. This supports reproducibility and experiment management.

The environment simulates a traffic network using the SUMO (Simulation of Urban MObility) simulator. The RL agent interacts with this environment by controlling traffic lights at intersections. Key parameters include:

- **State Space:** Each state represents the current traffic conditions at an intersection, such as queue lengths, waiting times, densities, and other features (e.g., green time, outflow rate, travel speed, travel time, and special events like accidents or pedestrian crossings).
- **Action Space:** The agent selects from a discrete set of possible traffic light phase changes and their durations, such as adjusting the green phase by a delta.
- **Reward Signal:** The environment provides feedback based on traffic efficiency metrics (e.g., reduced waiting time, improved flow), calculated as a weighted sum of normalized metrics.
- **Simulation Parameters:** Configurable via YAML, including simulation steps, memory buffer sizes, traffic light configurations, and agent hyperparameters.


## Agent Design and States

The agent state now includes DESRA hints and accident features for robust decision-making. The codebase is modularized and uses SKRL for agent management and training. See [MODULARIZATION_STATUS.md](MODULARIZATION_STATUS.md) for details.

The RL agent is implemented as a Deep Q-Network (DQN) with optional attention mechanisms ([SENet](https://link.springer.com/article/10.1007/s40747-025-01841-9)). The agent design includes:

- **States:** Encoded as vectors containing traffic features (green_time, outflow_rate, travel_speed, travel_time, density, and special events).
- **Actions:** Each action corresponds to a possible change in the traffic light phase and the change in duration of the green light.
- **Policy:** The agent uses an epsilon-greedy policy to balance exploration and exploitation during training.

## Reward System

The reward system is designed to encourage efficient traffic management:

- **Positive Rewards:** Given for reducing green_time, outflow_rate, travel_speed, travel_time, density, and for reacting safely to special events.
- **Penalties:** Applied for increased congestion, long green times, long waiting times, or unsafe phase switching.
- **Reward Calculation:** The reward is a weighted sum of normalized traffic efficiency metrics.

## RL Algorithm

The project uses a Deep Q-Network (DQN) algorithm with the following features:

- **Q-Learning:** The agent learns to estimate the expected cumulative reward (Q-value) for each action in a given state.
- **Neural Network:** The Q-function is approximated using a neural network with configurable layers and optional attention modules.
- **Experience Replay:** A replay buffer stores past experiences to break correlation and stabilize learning.
- **Target Network:** A separate target network is used to compute stable target Q-values.
- **Variations:** The implementation supports Double DQN logic and weighted loss functions for improved stability.

## Training and Evaluation Strategy

- **Training:** The agent is trained over multiple episodes, each consisting of several simulation steps. At each step, the agent selects actions, observes rewards, and updates its policy using mini-batch gradient descent.
- **Exploration vs. Exploitation:** An epsilon-greedy strategy is used, with epsilon decaying over time to shift from exploration to exploitation.
- **Evaluation:** Performance is evaluated by averaging key metrics (e.g., reward, travel time, queue length) over episodes and intersections. Plots and logs are generated for analysis.
- **Model Saving:** The trained model is saved periodically for later evaluation.

## Implementation Details

- **Programming Language:** Python 3.10+ (recommended)
- **Key Libraries:**
  - `torch`, `torchvision`, `torchaudio` (for deep learning)
  - `numpy` (for numerical operations)
  - `matplotlib` (for visualization)
  - `PyYAML` (for configuration)
  - `traci` (for SUMO interface)
  - `torchinfo` (for model summaries)
- **Simulation Tool:** [SUMO](https://www.eclipse.org/sumo/) (Simulation of Urban MObility)
- **Setup:**  
  - Install dependencies from `requirements.txt`
  - Set the `SUMO_HOME` environment variable to your SUMO installation path
  - Configure training parameters in the YAML config file

## Ethical and Safety Considerations

Deploying RL-based traffic signal control in real-world scenarios raises important ethical and safety concerns:

- **Safety:** The agent must never make decisions that could endanger human life, such as unsafe phase switching that risks collisions or fails to accommodate emergency vehicles and pedestrians.
- **Testing:** Extensive simulation testing are required before deployment. 
- **Transparency:** The decision-making process should be interpretable and auditable to ensure accountability.
- **Fairness:** The system should avoid introducing or amplifying biases that could unfairly disadvantage certain road users or neighborhoods.
- **Privacy:** Any use of real-world data must comply with privacy regulations and protect individual identities.

Careful consideration of these issues is essential to ensure that RL-STSC contributes positively to urban mobility and public safety.