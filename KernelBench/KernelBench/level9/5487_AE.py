import torch
from torch import nn
import torch.utils.data
import torch.utils.data.distributed


class AE(nn.Module):

    def __init__(self):
        super(AE, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=19, padding=9)
        self.conv2 = nn.Conv2d(16, 4, kernel_size=15, padding=7)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.t_conv1 = nn.ConvTranspose2d(4, 16, kernel_size=15, padding=7)
        self.t_conv2 = nn.ConvTranspose2d(16, 1, kernel_size=19, padding=9)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x, pool1 = self.pool(x)
        x = self.relu(self.conv2(x))
        x, pool2 = self.pool(x)
        x = self.unpool(x, pool2)
        x = self.relu(self.t_conv1(x))
        x = self.unpool(x, pool1)
        x = self.relu(self.t_conv2(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
