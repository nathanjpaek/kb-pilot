import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fc1 = nn.Linear(in_features, in_features // 2)
        self.fc2 = nn.Linear(in_features // 2, out_features)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        input = self.dropout(input)
        x = F.leaky_relu(self.fc1(input))
        x = self.fc2(x)
        return x

    def __repr__(self):
        return '{} ({} -> {})'.format(self.__class__.__name__, self.
            in_features, self.out_features)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
