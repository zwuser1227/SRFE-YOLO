import torch
import torch.nn as nn
import math
class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
        
class DynamicECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(DynamicECA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv1d(1, channel // 16, kernel_size=1, bias=False)
        self.fc2 = nn.Conv1d(channel // 16, 1, kernel_size=1, bias=False)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y_dynamic = self.fc1(y.squeeze(-1).transpose(-1, -2))
        y_dynamic = self.fc2(y_dynamic).transpose(-1, -2).unsqueeze(-1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = y * self.sigmoid(y_dynamic)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
