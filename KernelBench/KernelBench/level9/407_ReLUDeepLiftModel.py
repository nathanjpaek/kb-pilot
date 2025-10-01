import torch
import torch.nn as nn


class ReLUDeepLiftModel(nn.Module):
    """
        https://www.youtube.com/watch?v=f_iAM0NPwnM
    """

    def __init__(self):
        super().__init__()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x1, x2, x3=2):
        return 2 * self.relu1(x1) + x3 * self.relu2(x2 - 1.5)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
