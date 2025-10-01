import torch
import torch.nn as nn


class MyKernelTorch(nn.Module):

    def __init__(self, n_features: 'int'):
        super().__init__()
        self.dense1 = nn.Linear(n_features, 20)
        self.dense2 = nn.Linear(20, 2)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x = nn.ReLU()(self.dense1(x))
        return self.dense2(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
