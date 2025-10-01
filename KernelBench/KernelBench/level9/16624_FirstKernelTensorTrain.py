import torch
from torch import nn
import torch.nn.functional as F


class FirstKernelTensorTrain(nn.Module):

    def __init__(self, m, r_j):
        super(FirstKernelTensorTrain, self).__init__()
        self.fc1 = nn.Linear(m, r_j, bias=False)
        self.m = m
        self.r_j = r_j

    def forward(self, tensor):
        transformed_tensor = self.fc1(tensor)
        return F.relu(transformed_tensor)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'m': 4, 'r_j': 4}]
