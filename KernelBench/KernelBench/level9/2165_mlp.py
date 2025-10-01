import torch
from torch import nn


class mlp(nn.Module):

    def __init__(self, in_feature, **kwargs):
        super().__init__()
        self.in_feature = in_feature
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_feature, in_feature)
        self.dropout1 = nn.Dropout(p=0.3)
        self.linear2 = nn.Linear(in_feature, in_feature // 2)
        self.dropout2 = nn.Dropout(p=0.1)
        self.linear3 = nn.Linear(in_feature // 2, 1)

    def forward(self, input):
        x = self.linear1(input)
        x = self.dropout1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_feature': 4}]
