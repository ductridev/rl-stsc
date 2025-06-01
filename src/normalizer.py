class Normalizer:
    def __init__(self, min_val=float('inf'), max_val=float('-inf')):
        self.min_val = min_val
        self.max_val = max_val

    def update(self, val):
        if val < self.min_val:
            self.min_val = val
        if val > self.max_val:
            self.max_val = val

    def normalize(self, val):
        if self.max_val == self.min_val:
            return 0.0  # avoid divide by zero
        return (val - self.min_val) / (self.max_val - self.min_val)
