import os
import torch
import torch.nn as nn
import torch.nn.init
import torch.nn.functional as F
import math
import torch.fft as fft
from models.KCAN import KANBlock


# ----------------------------------- RCAN ------------------------------------------
## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            #
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):

    def __init__(self, n_feat, reduction, bn=False):
        super().__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(nn.ReLU(inplace=True))
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class RKAB(nn.Module):  # Residual KAN Attention Block
    def __init__(self, n_feat):
        super().__init__()
        self.kan_attn = KANBlock(n_feat)

    def forward(self, x):
        res = self.kan_attn(x)
        return x + res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, reduction):
        super().__init__()
        modules_body = [
            RCAB(n_feat, reduction, bn=False),  # 第1个是 RCAB
            RKAB(n_feat),                      # 第2个是 RKAB
            RCAB(n_feat, reduction, bn=False)  # 第3个是 RCAB
        ]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        return x + res


## Residual Channel Attention Network (RCAN)
class RCKAN(nn.Module):
    def __init__(self, n_resgroups=5, n_feats=48, reduction=16, n_channels=1, n_classes=1):
        super().__init__()
        # define head module
        modules_head = [nn.Conv2d(n_channels, n_feats, kernel_size=3, stride=1, padding=1, bias=True)]
        self.head = nn.Sequential(*modules_head)

        # define body module
        modules_body = [
            ResidualGroup(n_feats, reduction=reduction)
            for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # define tail module
        modules_tail = [nn.Conv2d(n_feats, n_classes, kernel_size=3, stride=1, padding=1, bias=True)]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    ## Residual Channel Attention Network (RCAN)


