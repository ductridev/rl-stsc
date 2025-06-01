class Normalizer:
    def __init__(self, min_val=1e10, max_val=-1e10):
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, val):
        self.min_val = min(self.min_val, val)
        self.max_val = max(self.max_val, val)

        if self.max_val == self.min_val:
            return 0.0  # avoid divide by zero
        
        norm = (val - self.min_val) / (self.max_val - self.min_val)  # Scale to range 0 to 1
        return norm * 2 - 1
