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

    def __init__(self, max_size, min_size):
        """Initialize the replay memory with a fixed capacity."""
        self.max_size = max_size
        self.min_size = min_size
        self.memory = []

    def push(self, state, action, green_time, reward, next_state, done=False):
        """
        Save a transition into the replay buffer.

        Args:
            state: current state
            action: action taken
            reward: reward received
            next_state: next state after the action
            done (bool): whether the episode has ended
        """
        self.memory.append(
            ((state, action, green_time), reward, (next_state, None), done)
        )
        if self.get_size() > self.max_size:
            self.memory.pop(
                0
            )  # if the length is greater than the size of memory, remove the oldest element

    def get_samples(self, batch_size):
        """
        Get a sample of size n from the replay buffer.

        Args:
            batch_size (int): number of transitions to sample

        Returns:
            A list of transitions sampled randomly
        """
        if self.get_size() < self.min_size:
            return []

        if batch_size > self.get_size():
            return random.sample(
                self.memory, self.get_size()
            )  # get all samples if batch_size is larger than the buffer size
        return random.sample(self.memory, batch_size)

    def get_size(self):
        """
        Get the current size of the replay buffer.

        Returns:
            int: current size of the replay buffer
        """
        return len(self.memory)

    def __len__(self):
        """Return the current size of the replay buffer."""
        return len(self.memory)
