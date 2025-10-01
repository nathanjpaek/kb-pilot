import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNAutoencoder(nn.Module):

    def __init__(self, depth_0=1, depth_1=64, depth_2=32, depth_3=16,
        lastdepth=1):
        super(CNNAutoencoder, self).__init__()
        self.depth_0 = depth_0
        self.depth_1 = depth_1
        self.depth_2 = depth_2
        self.depth_3 = depth_3
        self.enc1 = nn.Conv2d(self.depth_0, self.depth_1, kernel_size=3,
            stride=1, padding=1)
        self.enc2 = nn.Conv2d(self.depth_1, self.depth_2, kernel_size=3,
            stride=1, padding=1)
        self.enc3 = nn.Conv2d(self.depth_2, self.depth_3, kernel_size=3,
            stride=1, padding=1)
        self.ltnt = nn.Conv2d(self.depth_3, self.depth_3, kernel_size=3,
            stride=1, padding=1)
        self.dec3 = nn.ConvTranspose2d(self.depth_3, self.depth_2,
            kernel_size=1, stride=1)
        self.dec2 = nn.ConvTranspose2d(self.depth_2, self.depth_1,
            kernel_size=1, stride=1)
        self.dec1 = nn.ConvTranspose2d(self.depth_1, self.depth_0,
            kernel_size=1, stride=1)
        self.out = nn.Conv2d(self.depth_0, 1, 1)
        self.pool = nn.MaxPool2d(2, 2, return_indices=True, ceil_mode=False)
        self.unpool = nn.MaxUnpool2d(2, 2)

    def forward(self, x):
        x = F.relu(self.enc1(x))
        x, indx1 = self.pool(x)
        x = F.relu(self.enc2(x))
        x, indx2 = self.pool(x)
        x = F.relu(self.enc3(x))
        x, indx3 = self.pool(x)
        x = F.relu(self.ltnt(x))
        x = self.unpool(x, indx3, output_size=indx2.size())
        x = F.relu(self.dec3(x))
        x = self.unpool(x, indx2, output_size=indx1.size())
        x = F.relu(self.dec2(x))
        x = self.unpool(x, indx1)
        x = F.relu(self.dec1(x))
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
