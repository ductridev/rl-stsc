import random
from collections import defaultdict


class MemoryPalace:
    """
    Memory Palace for phase-action balanced experience replay.

    Each unique (phase, action) combination has its own memory buffer.
    """

    def __init__(self, max_size_per_palace, min_size_to_sample):
        """
        Args:
            max_size_per_palace (int): Max size for each individual memory buffer
            min_size_to_sample (int): Minimum size needed to sample from a palace
        """
        self.max_size = max_size_per_palace
        self.min_size = min_size_to_sample
        self.palaces = defaultdict(list)  # {(phase, action): [experiences]}

    def _key(self, action):
        """Define how to key the palace — here (green_time, action)"""
        return (action)

    def push(self, state, action, reward, next_state, done=False):
        """
        Save a transition into the corresponding memory palace.
        """
        key = self._key(action)
        buffer = self.palaces[key]
        buffer.append(((state, action), reward, (next_state, None), done))
        if len(buffer) > self.max_size:
            buffer.pop(0)  # Remove oldest item if buffer is full

    def get_balanced_samples(self, batch_size_per_palace):
        """
        Sample balanced data from all palaces.

        Returns:
            List of transitions sampled evenly from all buffers.
        """
        samples = []
        for key, buffer in self.palaces.items():
            if len(buffer) >= self.min_size:
                samples.extend(
                    random.sample(buffer, min(batch_size_per_palace, len(buffer)))
                )
        return samples

    def get_size(self):
        return sum(len(buf) for buf in self.palaces.values())

    def clean(self):
        self.palaces = defaultdict(list)

    def __len__(self):
        return self.get_size()
