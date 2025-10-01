import torch
import torch.utils.data
import torch.nn.functional as F
from torch import nn
import torch


class FCBottleNeck(nn.Module):

    def __init__(self, InFeatureSize):
        super().__init__()
        self.FC1 = nn.Linear(InFeatureSize, 2048)
        self.FC2 = nn.Linear(2048, 2048)
        self.FC3 = nn.Linear(2048, InFeatureSize)

    def forward(self, x):
        x_pe = x
        x_pe = F.relu(self.FC1(x_pe))
        x_pe = F.relu(self.FC2(x_pe))
        x_pe = self.FC3(x_pe)
        return x_pe


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'InFeatureSize': 4}]
