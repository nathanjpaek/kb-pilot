import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class CONV(nn.Module):

    def __init__(self, input_shape, device):
        super(CONV, self).__init__()
        self.device = device
        self.input_shape = input_shape
        self.poolavg = nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4,
            padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = self.poolavg(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        return x

    def get_last_layers(self):
        x = np.zeros(self.input_shape, dtype=np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x).float()
        res = self.forward(x)
        res = [int(x) for x in res[0].shape]
        return res[0]


def get_inputs():
    return [torch.rand([4, 4, 128, 128])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'device': 0}]
