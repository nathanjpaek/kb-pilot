import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):

    def __init__(self, individual_featured):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(individual_featured, 2 * individual_featured)
        self.w_2 = nn.Linear(2 * individual_featured, individual_featured)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'individual_featured': 4}]
