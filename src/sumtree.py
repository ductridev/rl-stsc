import numpy as np

class SumTree:
    """
    Prioritized Experience Replay Buffer using a SumTree data structure.
    Prioritized experience replay is a technique used in reinforcement learning to improve the efficiency of learning by sampling more important experiences more frequently.
    It assigns a priority to each experience based on its importance, and samples experiences based on their priorities.
    This allows the agent to learn more from experiences that are more informative or surprising.
    A SumTree is a binary tree where each parent node's value is the sum of its children. 
    The SumTree data structure is used to efficiently manage the priorities of experiences.
    A SumTree enables O(logN) sampling and updates of priorities.
    This is useful for implementing prioritized experience replay in reinforcement learning.
    Attributes:
        capacity (int): maximum number of experiences to store
        tree (np.ndarray): array representing the SumTree
        data (list): list to hold the actual experiences
        write (int): index for the next experience to be added
        n_entries (int): current number of experiences in the tree
    """
    def __init__(self, capacity):
        """
        Initialize the SumTree with a given capacity.

        Args:
            capacity (int): maximum number of experiences to store
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        """
        Propagate the change in priority up the tree.
        This is done by updating the parent nodes with the change in priority.
        Args:
            idx (int): index of the node to propagate from
            change (float): change in priority
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, priority):
        """
        Update the priority of a specific experience in the tree.
        This is done by updating the value at the given index and propagating the change up the tree.
        Args:
            idx (int): index of the experience to update
            priority (float): new priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def add(self, priority, data):
        """
        Add a new experience to the tree with a given priority.
        This is done by adding the experience to the data list and updating the tree with the priority.
        The experience is added at the next available index, and the tree is updated accordingly.
        Args:
            priority (float): priority of the experience
            data: actual experience data
        """
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, s):
        """
        Get an experience from the tree based on a given sample value.
        This is done by traversing the tree to find the experience with the highest priority that is less than or equal to the sample value.
        Args:
            s (float): sample value
        Returns:
            tuple: (index of the experience, priority of the experience, actual experience data)
        """
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx, self.tree[idx], self.data[idx - self.capacity + 1]
            else:
                if s <= self.tree[left]:
                    idx = left
                else:
                    s -= self.tree[left]
                    idx = right