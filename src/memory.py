import random
from collections import deque, namedtuple

class ReplayMemory:
    """
    Experience replay buffer for Deep Q-Networks.

    Attributes:
        capacity (int): maximum number of experiences to store
        memory (deque): buffer to hold experiences
        transition (namedtuple): structure to hold (state, action, reward, next_state, done)
    """

    def __init__(self, capacity):
        """Initialize the replay memory with a fixed capacity."""
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, state, action, reward, next_state, done):
        """
        Save a transition into the replay buffer.

        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state after the action
            done (bool): whether the episode has ended
        """
        self.memory.append(self.transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions for training.

        Args:
            batch_size (int): number of transitions to sample

        Returns:
            A list of transitions sampled randomly
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the replay buffer."""
        return len(self.memory)
