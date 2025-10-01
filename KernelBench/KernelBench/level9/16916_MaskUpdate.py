import torch
from torch import nn
from torch.nn.parameter import Parameter


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.updateFunc = nn.ReLU(False)
        self.alpha = Parameter(torch.tensor(alpha, dtype=torch.float32))

    def forward(self, inputMaskMap):
        self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
