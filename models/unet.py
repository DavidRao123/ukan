import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

LOW_IMG_HEIGHT = 64
LOW_IMG_WIDTH = 64

class FirstFeature(nn.Module):
    '''
    Implementation of UNET with Skip connections
    '''

    def __init__(self, in_channels, out_channels):
        super(FirstFeature, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class FinalOutput(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalOutput, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.conv(x)

class SR_Unet(nn.Module):
    def __init__(
            self, n_channels=1, n_classes=1
    ):
        super(SR_Unet, self).__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.resize_fnc = transforms.Resize((LOW_IMG_HEIGHT * 4, LOW_IMG_HEIGHT * 4),
        #                                     antialias=True)
        self.in_conv1 = FirstFeature(n_channels, 64)
        self.in_conv2 = ConvBlock(64, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = ConvBlock(1024 + 512, 512)
        self.dec_2 = ConvBlock(512 + 256, 256)
        self.dec_3 = ConvBlock(256 + 128, 128)
        self.dec_4 = ConvBlock(128 + 64, 64)
        self.out_conv = FinalOutput(64, n_classes)

    def forward(self, x):
        x_in = x
        # x = self.resize_fnc(x)
        x = self.in_conv1(x)
        x1 = self.in_conv2(x)

        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        if x.size(2) != x4.size(2) or x.size(3) != x4.size(3):
            x4 = center_crop(x4, x)
        x = torch.cat([x, x4], dim=1)
        x = self.dec_1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.size(2) != x3.size(2) or x.size(3) != x3.size(3):
            x3 = center_crop(x3, x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.size(2) != x2.size(2) or x.size(3) != x2.size(3):
            x2 = center_crop(x2, x)
        x = torch.cat([x, x2], dim=1)
        x = self.dec_3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if x.size(2) != x1.size(2) or x.size(3) != x1.size(3):
            x1 = center_crop(x1, x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_4(x)
        x = F.interpolate(x, size=x_in.shape[2:], mode='bilinear', align_corners=False)
        x = self.out_conv(x)
        return x

def center_crop(skip, target):
    _, _, h, w = target.size()
    _, _, hs, ws = skip.size()

    crop_top = (hs - h) // 2
    crop_left = (ws - w) // 2

    return skip[:, :, crop_top:crop_top + h, crop_left:crop_left + w]

if __name__ == '__main__':
    x = torch.rand(1, 1, 64, 64)

    # model = UNet().cuda()
    model = SR_Unet(n_channels=1)
    # model.eval()
    y = model(x)
    print('Output shape:', y.shape)