import random
import numpy as np
from sumtree import SumTree

class PERMemory:
    """
    Prioritized Experience Replay (PER) memory buffer.
    """
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.alpha = alpha  # controls how much prioritization is used

    def push(self, td_error, transition):
        priority = (abs(td_error) + 1e-6) ** self.alpha
        self.tree.add(priority, transition)

    def sample(self, batch_size, beta):
        segment = self.tree.tree[0] / batch_size
        batch, idxs, weights = [], [], []
        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i+1))
            idx, p, data = self.tree.get(s)
            prob = p / self.tree.tree[0]
            weight = (prob * self.tree.n_entries) ** (-beta)
            batch.append(data)
            idxs.append(idx)
            weights.append(weight)
        weights = np.array(weights) / max(weights)
        return batch, idxs, weights

    def update_priorities(self, idxs, td_errors):
        for idx, error in zip(idxs, td_errors):
            priority = (abs(error) + 1e-6) ** self.alpha
            self.tree.update(idx, priority)
