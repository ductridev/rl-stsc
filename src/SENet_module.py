import torch.nn as nn

class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()

        # Global pooling: compress each feature dimension into a single representative value
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers to learn channel-wise attention:
        # 1. Reduce feature size by 'reduction' to limit complexity and encourage sparsity
        # 2. Apply ReLU for non-linearity
        # 3. Restore back to original channel size
        # 4. Apply Sigmoid to get scaling weights between 0 and 1
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # Dimension reduction (compression)
            nn.ReLU(inplace=True),                                 # Non-linear activation
            nn.Linear(channel // reduction, channel, bias=False),  # Dimension restoration (expansion)
            nn.Sigmoid()                                           # Output scale factors (0,1) per channel
        )

    def forward(self, x):
        # Input shape: [batch_size, num_features]
        b, c = x.size()  # Extract batch size (b) and feature dimension (c)

        # Add a singleton dimension for pooling (required by AdaptiveAvgPool1d)
        x_unsqueezed = x.unsqueeze(2)  # Shape becomes: [batch_size, num_features, 1]

        # Apply average pooling across the sequence length (trivial here since length=1)
        pooled = self.avg_pool(x_unsqueezed)  # Output shape: [batch_size, num_features, 1]

        # Remove the last dimension to get back to [batch_size, num_features]
        squeezed = pooled.view(b, c)

        # Pass through fully connected layers to compute attention scaling factors
        scale = self.fc(squeezed)  # Output: [batch_size, num_features], values between 0 and 1

        # Apply scaling to original input (element-wise multiplication)
        recalibrated = x * scale  # Each feature is scaled according to learned importance

        return recalibrated  # Return recalibrated feature vector
