
## 🧠 Training Logic

**New Feature:** After each training run, a snapshot of the configuration used is automatically saved as `config_snapshot.yaml` (YAML format) in the results folder. This ensures reproducibility and easy experiment tracking.

**Modularization & SKRL Integration:** The training logic is now fully modularized, with SKRL agent management and factory patterns for environment/model creation. See [MODULARIZATION_STATUS.md](MODULARIZATION_STATUS.md) for details.

This section describes the core training logic behind the Deep Q-Network (DQN) implementation used in this project, focusing on batch training with a target network for stability.

### 🧾 Step-by-Step

#### 1. **Sample a Batch**
Each traffic light (intersection) maintains its own experience replay memory. For every training iteration, a batch of experiences is sampled:

```python
states, actions, rewards, next_states, dones = zip(*batch)
```

Each experience is a tuple: `(state, action, reward, next_state, done)`.

---

#### 2. **Compute Q-values for Current States**
We pass the current `states` batch through the main DQN to get the predicted Q-values:

```python
q_values = self.predict_batch(states, output_dim)
q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
```

- `q_values`: full Q-value distribution for each state
- `q_value`: the predicted Q-value for the action that was actually taken

---

#### 3. **Compute Q-values for Next States Using Target Network**
We use a separate `target_net` (which is a copy of the main network) to compute Q-values for the next states:

```python
next_q_values = target_net.predict_batch(next_states, output_dim)
max_next_q_values = torch.max(next_q_values, dim=1)[0]
```

This ensures that the target used for training is stable and doesn't change every step.

##### 🎯 Why We Use a Target Network in DQN?

Deep Q-Networks (DQN) estimate the optimal action-value function using neural networks. A common issue in DQN is **instability or divergence during training**. This is because the same network is used to calculate both:

- The **predicted Q-value**: `Q(s, a)`
- The **target Q-value**: `r + γ * max_a' Q(s', a')`

##### ❗ Problem: Moving Targets

Using the same network to estimate both the prediction and the target causes the target to shift every time the network updates. This leads to unstable learning and makes convergence harder.

---

##### ✅ Solution: Target Network

A **Target Network** is a separate copy of the Q-network that is used to compute the target Q-value. It is **updated less frequently** (e.g., every N steps or slowly using soft updates). This stabilizes learning by keeping the target fixed for a while.

In `src/simulation.py`:

```python
if episode % 10 == 0:
    self.target_net.load_state_dict(self.agent.state_dict())
```

---

#### 4. **Calculate Target Q-value**
We calculate the target Q-value using the Bellman equation:

```python
target_q_value = rewards + (1 - dones) * gamma * max_next_q_values
```

- If the episode is done (`done = 1`), we only use the immediate reward.
- Otherwise, we include the discounted future reward.

---

#### 5. **Compute Loss and Update**
We compute the loss between the predicted Q-value and the target, then perform a gradient descent step:

```python
loss = F.mse_loss(q_value, target_q_value.detach())
loss.backward()
optimizer.step()
```

This adjusts the DQN to better approximate the expected returns.

---

#### 6. **Periodically Update the Target Network**
Describe in [🎯 Why We Use a Target Network in DQN?](#-why-we-use-a-target-network-in-dqn) Section

---
