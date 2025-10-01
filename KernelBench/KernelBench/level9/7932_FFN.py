import torch
from torch import nn as nn
import torch.nn.functional as F


class FFN(nn.Module):

    def __init__(self, d_model, hidden_size=1024):
        super().__init__()
        self.ln1 = nn.Linear(d_model, hidden_size)
        self.ln2 = nn.Linear(hidden_size, d_model)

    def reset_params(self):
        nn.init.xavier_normal_(self.ln1.weight.data)
        nn.init.xavier_normal_(self.ln2.weight.data)
        nn.init.constant_(self.ln1.bias.data, 0)
        nn.init.constant_(self.ln2.bias.data, 0)

    def forward(self, x, x_mas):
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return x * x_mas


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4}]
