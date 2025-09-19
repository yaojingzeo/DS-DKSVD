"""
Channel & space attention module
"""
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    def __init__(self, in_channel, reduction):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel,  reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, in_channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avgOut = self.fc(self.avg_pool(x).view(b, c))  #avg_pool:[N, 64, 1, 1]  avg:[N, 64]
        maxOut = self.fc(self.max_pool(x).view(b, c))  #maxOut:[N, 64]
        y = self.sigmoid(avgOut + maxOut).view(b, c, 1, 1)  #y:[N, 64, 1, 1]
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):  
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding= (kernel_size -1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, _, h, w = x.size()
        avgOut = torch.mean(x, dim=1, keepdim=True)  #avgOut[N, 1, 16, 16]
        maxOut, _ = torch.max(x, dim=1, keepdim=True)  #maxOut[N, 1, 16, 16]
        y = torch.cat([avgOut, maxOut], dim=1)  #y[N, 2, 16, 16]
        y = self.sigmoid(self.conv(y))  #y[N, 1, 16, 16]
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channel, reduction, kernel_size=1):
        super(CBAM, self).__init__()
        self.ChannelAtt = ChannelAttention(channel, reduction)
        self.SpatialAtt = SpatialAttention(kernel_size) 

    def forward(self, x):
        x = self.ChannelAtt(x)
        x = self.SpatialAtt(x)
        return x