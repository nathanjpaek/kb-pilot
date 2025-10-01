import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageProcessingModuleAlt(nn.Module):

    def __init__(self, n_filters):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=n_filters * 2,
            kernel_size=7)
        self.conv2 = nn.Conv2d(in_channels=n_filters * 2, out_channels=
            n_filters, kernel_size=7)
        self.conv3 = nn.Conv2d(in_channels=n_filters, out_channels=
            n_filters // 2, kernel_size=7)

    def forward(self, observation):
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        return observation


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {'n_filters': 4}]
