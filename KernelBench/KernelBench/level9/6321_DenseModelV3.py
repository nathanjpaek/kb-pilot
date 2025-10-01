import torch
import torch.nn as nn


class DenseModelV3(nn.Module):

    def __init__(self, input_dim, num_classes=2):
        super(DenseModelV3, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, 2000)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(2000, 2000)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(2000, 2000)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(2000, 2000)
        self.relu5 = nn.ReLU(inplace=True)
        self.fc6 = nn.Linear(2000, 2000)
        self.relu6 = nn.ReLU(inplace=True)
        self.fc7 = nn.Linear(2000, 2000)
        self.relu7 = nn.ReLU(inplace=True)
        self.fc8 = nn.Linear(2000, 2000)
        self.relu8 = nn.ReLU(inplace=True)
        self.fc9 = nn.Linear(2000, 400)
        self.relu9 = nn.ReLU(inplace=True)
        if num_classes == 2:
            self.fc10 = nn.Linear(400, 1)
        else:
            self.fc10 = nn.Linear(400, num_classes)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu4(self.fc4(x))
        x = self.relu5(self.fc5(x))
        x = self.relu6(self.fc6(x))
        x = self.relu7(self.fc7(x))
        x = self.relu8(self.fc8(x))
        x = self.relu9(self.fc9(x))
        x = self.fc10(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
