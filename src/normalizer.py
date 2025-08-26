import math

class Normalizer:
    def __init__(self, scaling_factor=10.0):
        """
        Initialize the Normalizer with running statistics for z-score normalization.
        
        Args:
            scaling_factor (float): Multiplier applied to the z-score. 
                                  Default is 10.0 for amplified sensitivity.
                                  Use 1.0 for standard z-score normalization.
        """
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.std = 1.0  # Initialize to 1.0 to avoid division by zero initially
        self.scaling_factor = scaling_factor

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
        
        # Return z-score normalized value with scaling factor
        if self.std == 0.0:
            return 0.0  # avoid divide by zero when all values are identical
        
        return self.scaling_factor * (val - self.mean) / self.std
