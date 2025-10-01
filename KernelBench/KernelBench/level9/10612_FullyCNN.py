import torch
from torch import nn


class FullyCNN(nn.Module):
    """UNET Without concatenation during decoding"""

    def __init__(self):
        super(FullyCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,
            stride=1, padding=1, padding_mode='reflect')
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=
            3, stride=1, padding=1, padding_mode='reflect')
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size
            =3, stride=1, padding=1, padding_mode='reflect')
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=24, out_channels=12, kernel_size
            =3, stride=1, padding=1, padding_mode='reflect')
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(in_channels=12, out_channels=6, kernel_size=
            3, stride=1, padding=1, padding_mode='reflect')
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3,
            stride=1, padding=1, padding_mode='reflect')
        self.relu6 = nn.ReLU()
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3,
            stride=1, padding=1, padding_mode='reflect')
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.relu5(self.conv5(x))
        x = self.relu6(self.conv6(x))
        x = self.relu7(self.conv7(x))
        return x


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
