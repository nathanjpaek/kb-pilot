import torch
from torch import nn
from torch.nn import functional as F


class MLP(torch.nn.Module):
    """MLP for patch segmentation."""

    def __init__(self, n_classes, input_dim):
        super().__init__()
        self.layer_1 = nn.Linear(input_dim, 200)
        self.layer_2 = nn.Linear(200, 100)
        self.layer_3 = nn.Linear(100, n_classes)

    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_classes': 4, 'input_dim': 4}]
