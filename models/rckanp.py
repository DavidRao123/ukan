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


## RKAB
class RKAB(nn.Module):  # Residual KAN Attention Block
    def __init__(self, n_feat):
        super().__init__()
        self.kan_attn = KANBlock(n_feat)

    def forward(self, x):
        res = self.kan_attn(x)
        return x + res

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


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, reduction):
        super().__init__()
        self.rkab = RKAB(n_feat)
        self.rcab = RCAB(n_feat, reduction)


    def forward(self, x):
        cab = self.rcab(x)
        kab = self.rkab(x)
        res = cab + kab
        return res


## Residual Channel Attention Network (RCAN)
class RCKANP(nn.Module):
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

if __name__ == '__main__':
    # x = Variable(torch.rand(2,1,64,64)).cuda()
    x = torch.rand(1, 1, 128, 128)

    # model = UNet().cuda()
    model = RCKANP(n_channels=1)
    # model.eval()
    y = model(x)
    print('Output shape:', y.shape)