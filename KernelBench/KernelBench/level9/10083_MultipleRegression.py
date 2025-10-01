import torch
import torch.nn as nn


class MultipleRegression(nn.Module):

    def __init__(self, num_features):
        super(MultipleRegression, self).__init__()
        self.fc1 = nn.Linear(num_features, 64)
        self.fc2 = nn.Linear(64, 128)
        self.output = nn.Linear(128, 1)
        self.act = nn.Sigmoid()

    def forward(self, inputs):
        x = self.act(self.fc1(inputs))
        x = self.act(self.fc2(x))
        x = self.output(x)
        return x

    def predict(self, test_inputs):
        x = self.act(self.fc1(test_inputs))
        x = self.act(self.fc2(x))
        x = self.output(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
