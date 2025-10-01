import torch
import torch.nn as nn


class FC1(nn.Module):

    def __init__(self, nInput, activate, weight):
        super(FC1, self).__init__()
        self.nInput = nInput
        self.fc1 = nn.Linear(self.nInput, self.nInput * 2)
        self.fc2 = nn.Linear(self.nInput * 2, self.nInput)
        self.fc3 = nn.Linear(self.nInput, self.nInput // 2)
        self.fc4 = nn.Linear(self.nInput // 2, 1)
        self.weight = weight
        if activate == 'sigmoid':
            self.activate = nn.Sigmoid()
        else:
            self.activate = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.activate(self.weight * self.dropout(self.fc1(x)))
        x2 = self.activate(self.weight * self.dropout(self.fc2(x1)))
        x3 = self.activate(self.weight * self.fc3(x2))
        x4 = self.sigmoid(self.weight * self.fc4(x3))
        return x1, x2, x3, x4


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nInput': 4, 'activate': 4, 'weight': 4}]
