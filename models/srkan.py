from torch import nn
import torch
from models.archs import PatchEmbed, KANBlock


class SRKAN(nn.Module):
    def __init__(self, n_channels=1, n_classes= 1,  img_size=256):
        super(SRKAN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=9, padding=9 // 2)
        # Using patch_size=1 ensures each spatial location is tokenized individually.
        self.patch_embed = PatchEmbed(img_size= img_size, patch_size=1, stride=1, in_chans=64, embed_dim=64)
        self.kan_block = KANBlock(dim=64, drop=0., drop_path=0.)

        self.conv3 = nn.Conv2d(64, n_classes, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Step 1: CNN-based feature extraction
        x = self.relu(self.conv1(x))

        # Step 2: Tokenization via PatchEmbed.
        # The PatchEmbed converts the spatial feature map to tokens.
        # It returns a tensor of shape (B, N, 64) where N = H * W (256 * 256 = 65536)
        x, H, W = self.patch_embed(x)

        # Step 3: Process tokens through the KANBlock.
        x = self.kan_block(x, H, W)

        # Step 4: Reshape tokens back into a 2D spatial grid.
        B, N, C = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()  # Now shape: (B, 64, 256, 256)

        # Step 5: Reconstruction: Final convolution to generate the high-resolution output.
        x = self.conv3(x)
        return x

if __name__ == '__main__':
    # x = Variable(torch.rand(2,1,64,64)).cuda()
    x = torch.rand(1, 1, 128, 128)

    # model = UNet().cuda()
    model = SRKAN(n_channels=1)
    # model.eval()
    y = model(x)
    print('Output shape:', y.shape)