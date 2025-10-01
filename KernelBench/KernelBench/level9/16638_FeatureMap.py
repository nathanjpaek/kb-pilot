import torch
from torch import nn
import torch.nn.functional as F


class FeatureMap(nn.Module):

    def __init__(self, n, m, amount_of_division, batch_size):
        super(FeatureMap, self).__init__()
        self.m = m
        self.n = n
        self.amount_of_division = amount_of_division
        self.batch_size = batch_size
        self.fc1 = nn.Linear(self.n, self.m)

    def forward(self, tensor):
        last_dim = tensor.size()[-1]
        tensor = tensor.contiguous()
        tensor_reshaped = tensor.view(-1, last_dim)
        tensor_transformed = F.relu(self.fc1(tensor_reshaped))
        return tensor_transformed


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n': 4, 'm': 4, 'amount_of_division': 4, 'batch_size': 4}]
