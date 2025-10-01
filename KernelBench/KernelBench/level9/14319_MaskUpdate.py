import torch
import torch.nn as nn


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.updateFunc = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, inputMaskMap):
        """ self.alpha.data = torch.clamp(self.alpha.data, 0.6, 0.8)
        print(self.alpha) """
        return torch.pow(self.updateFunc(inputMaskMap), self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
