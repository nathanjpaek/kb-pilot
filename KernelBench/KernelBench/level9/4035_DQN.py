import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, num_in_features, num_out_features):
        super(DQN, self).__init__()
        self.linear1 = nn.Linear(num_in_features, 32)
        self.ln1 = nn.LayerNorm(32)
        self.linear2 = nn.Linear(32, 64)
        self.ln2 = nn.LayerNorm(64)
        self.linear3 = nn.Linear(64, 64)
        self.ln3 = nn.LayerNorm(64)
        self.linear4 = nn.Linear(64, 32)
        self.ln4 = nn.LayerNorm(32)
        self.out_layer = nn.Linear(32, num_out_features)

    def forward(self, x):
        x = F.leaky_relu(self.ln1(self.linear1(x)))
        x = F.leaky_relu(self.ln2(self.linear2(x)))
        x = F.leaky_relu(self.ln3(self.linear3(x)))
        x = F.leaky_relu(self.ln4(self.linear4(x)))
        return self.out_layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_in_features': 4, 'num_out_features': 4}]
