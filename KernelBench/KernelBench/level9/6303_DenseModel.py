import torch
import torch.nn as nn


class DenseModel(nn.Module):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(400, 400)
        self.relu2 = nn.ReLU(inplace=True)
        if num_classes == 2:
            self.fc3 = nn.Linear(400, 1)
        else:
            self.fc3 = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
