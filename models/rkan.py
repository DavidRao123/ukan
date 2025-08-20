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
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1)
        self.kan_attn = KANBlock(n_feat)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.kan_attn(res)
        return x + res


## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, n_feat, reduction, n_resblocks):
        super().__init__()
        modules_body = [
            RKAB(n_feat)
            for _ in range(n_resblocks)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


## Residual Channel Attention Network (RCAN)
class RKAN(nn.Module):
    def __init__(self, n_resblocks=3, n_resgroups=5, n_feats=48, reduction=16, n_channels=1, n_classes=1):
        super().__init__()
        # define head module
        modules_head = [nn.Conv2d(n_channels, n_feats, kernel_size=3, stride=1, padding=1, bias=True)]
        self.head = nn.Sequential(*modules_head)

        # define body module
        modules_body = [
            ResidualGroup(n_feats, reduction=reduction, n_resblocks=n_resblocks)
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


class RKANs(nn.Module):
    def __init__(self, n_resblocks=3, n_resgroups=5, n_feats=48, reduction=16, n_channels=3, n_classes=1):
        super().__init__()
        # define head module
        modules_head = [nn.Conv2d(n_channels, n_feats, kernel_size=3, stride=1, padding=1, bias=True)]
        self.head = nn.Sequential(*modules_head)

        # define body module
        modules_body = [
            ResidualGroup(n_feats, reduction=16, n_resblocks=n_resblocks)
            for _ in range(n_resgroups)]
        modules_body.append(nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1, bias=True))
        self.body = nn.Sequential(*modules_body)

        # define tail module
        modules_tail = [nn.Conv2d(n_feats, n_classes, kernel_size=3, stride=1, padding=1, bias=True), nn.Sigmoid()]
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x):
        x = self.head(x)
        res = self.body(x)
        res += x
        x = self.tail(res)
        return x

    ## RCAN downsize=3


class RKAN_D3(nn.Module):
    def __init__(self, n_resblocks=2, n_resgroups=5, n_feats=48, reduction=16, n_channels=3, n_classes=1):
        super().__init__()
        # define head module
        self.head = nn.Sequential(
            nn.Conv2d(n_channels, n_feats, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding='same'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        # define body module
        modules_body = [
            ResidualGroup(n_feats, reduction=16, n_resblocks=n_resblocks)
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
    model = RKAN(n_channels=1)
    # model.eval()
    y = model(x)
    print('Output shape:', y.shape)