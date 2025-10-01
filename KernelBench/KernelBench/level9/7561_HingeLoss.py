import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions
import torch.utils.data


class HingeLoss(nn.Module):

    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output):
        return F.relu(self.margin - output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
