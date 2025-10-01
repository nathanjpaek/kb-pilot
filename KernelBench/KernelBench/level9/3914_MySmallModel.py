import torch
import torch.nn as nn


class MySmallModel(nn.Module):

    def __init__(self, nodes):
        super().__init__()
        hidden_nodes = nodes * 2
        self.fc1 = nn.Linear(nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, nodes)
        self.fc3 = nn.Linear(nodes, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nodes': 4}]
