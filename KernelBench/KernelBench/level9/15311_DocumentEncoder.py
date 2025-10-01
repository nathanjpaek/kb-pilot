import torch
from torch import nn
import torch.nn.functional as F


class DocumentEncoder(nn.Module):

    def __init__(self, input_size, hidden_layer_sizes=(100,), activation=(
        'relu',), solver='adam'):
        super(DocumentEncoder, self).__init__()
        self.fc1 = nn.Linear(input_size, 12)
        self.fc2 = nn.Linear(12, 8)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.relu(self.fc2(out))
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
