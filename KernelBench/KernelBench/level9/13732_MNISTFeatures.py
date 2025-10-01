import torch
import torch.nn.functional as F
import torch.nn as nn


class MNISTFeatures(nn.Module):
    """
    A small convnet for extracting features
    from MNIST.
    """

    def __init__(self):
        """ """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 48, 5, 1)
        self.fc = nn.Identity()

    def forward(self, x):
        """ """
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, start_dim=1)
        return self.fc(x)


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
