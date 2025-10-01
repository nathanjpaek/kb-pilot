import torch
import torch.nn as nn


class FFN(nn.Module):

    def __init__(self, input_dim, num_class):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, num_class)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(torch.relu(self.layer1(x)))
        x = self.dropout(torch.relu(self.layer2(x)))
        x = self.dropout(torch.relu(self.layer3(x)))
        x = self.out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'num_class': 4}]
