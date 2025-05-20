from collections import deque
from memory import ReplayMemory

class NStepReplayMemory(ReplayMemory):
    """
    N-step experience replay buffer for Deep Q-Networks
    This is useful for implementing n-step Q-learning, where the agent learns from
    multiple steps of experience at once.
    """
    def __init__(self, capacity, n_step, gamma):
        """
        Initialize the n-step replay memory with a fixed capacity.
        Args:
            capacity (int): maximum number of experiences to store
            n_step (int): number of steps to consider for n-step Q-learning
            gamma (float): discount factor for future rewards
        """
        super().__init__(capacity)
        self.n_step = n_step
        self.gamma = gamma
        self.n_buffer = deque(maxlen=n_step)

    def push(self, state, action, reward, next_state, done = False):
        """
        Save a transition into the n-step replay buffer.
        This method accumulates rewards over n steps and stores the transition
        with the discounted return.
        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state after the action
            done (bool): whether the episode has ended
        """
        self.n_buffer.append((state, action, reward))
        if len(self.n_buffer) == self.n_step:
            R = sum(self.gamma ** i * t[2] for i, t in enumerate(self.n_buffer))
            s0, a0, _ = self.n_buffer[0]
            self.memory.append((s0, a0, R, next_state, done))
