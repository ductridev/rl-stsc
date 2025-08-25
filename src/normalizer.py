import math

class Normalizer:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.std = 1.0  # Initialize to 1.0 to avoid division by zero initially

    def normalize(self, val):
        # Update running statistics
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.sum_squares += delta * delta2
        
        # Calculate standard deviation (use sample std with n-1 denominator)
        if self.count > 1:
            variance = self.sum_squares / (self.count - 1)
            self.std = math.sqrt(variance)
        else:
            self.std = 1.0  # Default to 1.0 for single observation
        
        # Return z-score normalized value
        if self.std == 0.0:
            return 0.0  # avoid divide by zero when all values are identical
        
        return 10 * (val - self.mean) / self.std
