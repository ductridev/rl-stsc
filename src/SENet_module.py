import torch.nn as nn
class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [B, C]
        x = x.unsqueeze(2)  # [B, C, 1]
        y = self.avg_pool(x).squeeze(2)  # [B, C]
        y = self.fc(y)  # [B, C]
        return x.squeeze(2) * y
