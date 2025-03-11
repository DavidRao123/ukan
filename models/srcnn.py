from torch import nn
import torch

class SRCNN(nn.Module):
    def __init__(self, n_channels=1, n_classes= 1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, n_classes, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

if __name__ == '__main__':
    # x = Variable(torch.rand(2,1,64,64)).cuda()
    x = torch.rand(1, 1, 128, 128)

    # model = UNet().cuda()
    model = SRCNN(n_channels=1)
    # model.eval()
    y = model(x)
    print('Output shape:', y.shape)