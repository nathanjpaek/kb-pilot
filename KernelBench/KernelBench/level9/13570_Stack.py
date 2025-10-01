import torch
from torch import nn
from typing import *


class Stack(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, modalities):
        flattened = []
        for modality in modalities:
            flattened.append(torch.flatten(modality, start_dim=1))
        return torch.stack(flattened, dim=2)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
