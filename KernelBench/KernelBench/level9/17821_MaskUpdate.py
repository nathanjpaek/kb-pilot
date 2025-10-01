import torch
import torch.nn as nn
import torch.multiprocessing


class MaskUpdate(nn.Module):

    def __init__(self, alpha):
        super(MaskUpdate, self).__init__()
        self.func = nn.ReLU(True)
        self.alpha = alpha

    def forward(self, input_masks):
        return torch.pow(self.func(input_masks), self.alpha)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4}]
