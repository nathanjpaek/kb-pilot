import torch
import torch.nn as nn


class FCN_mse(nn.Module):
    """
    Predict whether pixels are part of the object or the background.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        c1 = torch.tanh(self.conv1(x))
        c2 = torch.tanh(self.conv2(c1))
        score = self.classifier(c2)
        return score


def get_inputs():
    return [torch.rand([4, 3, 64, 64])]


def get_init_inputs():
    return [[], {}]
