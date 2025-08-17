"""
Squeeze-and-Excitation Networks (SENet) implementation.

Based on the paper: "Squeeze-and-Excitation Networks" by Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu
https://arxiv.org/abs/1709.01507

This implementation provides both 1D and 2D versions for different input types.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D features (fully connected layers).
    
    This is suitable for feature vectors from fully connected networks.
    """
    
    def __init__(self, channel, reduction=16):
        """
        Args:
            channel (int): Number of input channels/features
            reduction (int): Channel reduction ratio for the bottleneck (default: 16)
        """
        super(SEBlock1D, self).__init__()
        self.channel = channel
        self.reduction = max(1, reduction)  # Ensure reduction is at least 1
        
        # Calculate reduced dimension, ensuring it's at least 1
        reduced_channels = max(1, channel // self.reduction)
        
        # Squeeze operation: Global average pooling (for 1D, this is just mean)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation operation: Two fully connected layers with ReLU and Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass of SE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels] or [batch_size, channels, 1]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        batch_size, channels = x.shape[:2]
        
        # Handle both [B, C] and [B, C, 1] inputs
        if x.dim() == 2:
            # For [B, C] input, add dimension for pooling
            y = x.unsqueeze(-1)  # [B, C, 1]
        else:
            y = x
        
        # Squeeze: Global average pooling
        y = self.avg_pool(y)  # [B, C, 1]
        y = y.view(batch_size, channels)  # [B, C]
        
        # Excitation: FC layers
        y = self.fc(y)  # [B, C]
        
        # Scale original input
        if x.dim() == 2:
            return x * y  # [B, C]
        else:
            return x * y.unsqueeze(-1)  # [B, C, 1]


class SEBlock2D(nn.Module):
    """
    Squeeze-and-Excitation Block for 2D features (convolutional layers).
    
    This is the original SENet block for convolutional neural networks.
    """
    
    def __init__(self, channel, reduction=16):
        """
        Args:
            channel (int): Number of input channels
            reduction (int): Channel reduction ratio for the bottleneck (default: 16)
        """
        super(SEBlock2D, self).__init__()
        self.channel = channel
        self.reduction = max(1, reduction)
        
        # Calculate reduced dimension
        reduced_channels = max(1, channel // self.reduction)
        
        # Squeeze operation: Global average pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Excitation operation: Two fully connected layers with ReLU and Sigmoid
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass of SE block.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        batch_size, channels, _, _ = x.size()
        
        # Squeeze: Global average pooling
        y = self.avg_pool(x).view(batch_size, channels)  # [B, C]
        
        # Excitation: FC layers
        y = self.fc(y).view(batch_size, channels, 1, 1)  # [B, C, 1, 1]
        
        # Scale original input
        return x * y.expand_as(x)


class SENet(nn.Module):
    """
    Adaptive SENet that works with both 1D and 2D inputs.
    
    This is the main class that should be used in your Q-network.
    It automatically detects input dimensions and applies appropriate SE attention.
    
    Backward compatible with the original implementation.
    """
    
    def __init__(self, channel, reduction=16, force_1d=True):
        """
        Args:
            channel (int): Number of input channels/features
            reduction (int): Channel reduction ratio (default: 16)
            force_1d (bool): Force 1D SE block for FC layers (default: True for Q-network compatibility)
        """
        super(SENet, self).__init__()
        self.channel = channel
        self.reduction = max(1, reduction)
        self.force_1d = force_1d
        
        # Calculate reduced dimension
        reduced_channels = max(1, channel // self.reduction)
        
        if force_1d:
            # Use 1D version for Q-network compatibility
            self.se_block = SEBlock1D(channel, self.reduction)
        else:
            # Create both blocks, will choose at runtime based on input
            self.se_1d = SEBlock1D(channel, self.reduction)
            self.se_2d = SEBlock2D(channel, self.reduction)
    
    def forward(self, x):
        """
        Forward pass that adaptively chooses SE block based on input dimensions.
        
        Args:
            x (torch.Tensor): Input tensor of various shapes:
                             - [batch_size, channels] -> Use SE1D
                             - [batch_size, channels, length] -> Use SE1D  
                             - [batch_size, channels, height, width] -> Use SE2D
                             
        Returns:
            torch.Tensor: Output tensor with same shape as input
        """
        if self.force_1d:
            return self.se_block(x)
        else:
            # Adaptive selection based on input dimensions
            if x.dim() <= 3:  # [B, C] or [B, C, L]
                return self.se_1d(x)
            else:  # [B, C, H, W] or higher
                return self.se_2d(x)


class SENetFC(nn.Module):
    """
    SENet specifically designed for fully connected layers in Q-networks.
    
    This version is optimized for the feature vectors typically found in 
    value function approximation networks.
    """
    
    def __init__(self, feature_dim, reduction=16, use_layer_norm=False):
        """
        Args:
            feature_dim (int): Dimension of feature vector
            reduction (int): Reduction ratio for bottleneck
            use_layer_norm (bool): Whether to apply layer normalization
        """
        super(SENetFC, self).__init__()
        self.feature_dim = feature_dim
        
        # Calculate reduced dimension, ensuring it's at least 1
        reduced_dim = max(1, feature_dim // max(1, reduction))
        
        # Squeeze and Excitation for FC features
        self.se_block = nn.Sequential(
            nn.Linear(feature_dim, reduced_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, feature_dim, bias=False),
            nn.Sigmoid()
        )
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(feature_dim) if use_layer_norm else None
    
    def forward(self, x):
        """
        Forward pass for fully connected features.
        
        Args:
            x (torch.Tensor): Input features [batch_size, feature_dim]
            
        Returns:
            torch.Tensor: Attention-weighted features [batch_size, feature_dim]
        """
        # Compute attention weights
        attention = self.se_block(x)  # [B, feature_dim]
        
        # Apply attention
        out = x * attention
        
        # Optional layer normalization
        if self.layer_norm is not None:
            out = self.layer_norm(out)
        
        return out


# For backward compatibility
SEBlock = SENet  # Alias for the main adaptive SE block


def test_senet():
    """Test function to verify SENet implementations work correctly."""
    print("Testing SENet implementations...")
    
    # Test 1D version (for FC layers)
    batch_size, channels = 32, 256
    x_1d = torch.randn(batch_size, channels)
    
    se_1d = SEBlock1D(channels)
    out_1d = se_1d(x_1d)
    print(f"SE1D: Input {x_1d.shape} -> Output {out_1d.shape}")
    assert out_1d.shape == x_1d.shape, "Shape mismatch in SE1D"
    
    # Test 2D version (for conv layers)
    x_2d = torch.randn(batch_size, channels, 8, 8)
    
    se_2d = SEBlock2D(channels)
    out_2d = se_2d(x_2d)
    print(f"SE2D: Input {x_2d.shape} -> Output {out_2d.shape}")
    assert out_2d.shape == x_2d.shape, "Shape mismatch in SE2D"
    
    # Test adaptive version (default force_1d=True)
    se_adaptive = SENet(channels)
    out_adaptive_1d = se_adaptive(x_1d)
    print(f"SENet (force_1d=True): Input {x_1d.shape} -> Output {out_adaptive_1d.shape}")
    assert out_adaptive_1d.shape == x_1d.shape, "Shape mismatch in SENet adaptive 1D"
    
    # Test adaptive version with auto-detection
    se_auto = SENet(channels, force_1d=False)
    out_auto_1d = se_auto(x_1d)
    out_auto_2d = se_auto(x_2d)
    print(f"SENet (auto-detect) 1D: Input {x_1d.shape} -> Output {out_auto_1d.shape}")
    print(f"SENet (auto-detect) 2D: Input {x_2d.shape} -> Output {out_auto_2d.shape}")
    assert out_auto_1d.shape == x_1d.shape, "Shape mismatch in SENet auto 1D"
    assert out_auto_2d.shape == x_2d.shape, "Shape mismatch in SENet auto 2D"
    
    # Test FC version
    se_fc = SENetFC(channels)
    out_fc = se_fc(x_1d)
    print(f"SENetFC: Input {x_1d.shape} -> Output {out_fc.shape}")
    assert out_fc.shape == x_1d.shape, "Shape mismatch in SENetFC"
    
    # Test edge case: small channel number
    small_channels = 4
    x_small = torch.randn(batch_size, small_channels)
    se_small = SENet(small_channels, reduction=16)  # reduction > channels
    out_small = se_small(x_small)
    print(f"SENet (small channels): Input {x_small.shape} -> Output {out_small.shape}")
    assert out_small.shape == x_small.shape, "Shape mismatch in small channels test"
    
    print("✅ All SENet tests passed!")


if __name__ == "__main__":
    test_senet()
