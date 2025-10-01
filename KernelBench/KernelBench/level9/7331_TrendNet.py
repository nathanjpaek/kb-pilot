import torch
import torch.nn as nn


class TrendNet(nn.Module):

    def __init__(self, feature_size):
        super(TrendNet, self).__init__()
        self.hidden_size1 = 16
        self.hidden_size2 = 16
        self.output_size = 1
        self.fc1 = nn.Linear(feature_size, self.hidden_size1)
        self.fc2 = nn.Linear(self.hidden_size1, self.hidden_size2)
        self.fc3 = nn.Linear(self.hidden_size2, self.output_size)

    def forward(self, input_data):
        input_data = input_data.transpose(0, 1)
        xout = self.fc1(input_data)
        xout = self.fc2(xout)
        xout = self.fc3(xout)
        return xout


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4}]
