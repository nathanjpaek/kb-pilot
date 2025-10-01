from torch.nn import Module
import torch
from torch.nn import Linear
import torch.nn as nn
import torch.nn.functional as F


class Model(Module):

    def __init__(self, input_shape, nb_classes, *args, **kwargs):
        super(Model, self).__init__()
        self.fc1 = Linear(input_shape[0], 25)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = Linear(25, 75)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = Linear(75, 200)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = Linear(200, nb_classes)
        self.dropout4 = nn.Dropout(0.3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.dropout4(x)
        x = F.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': [4, 4], 'nb_classes': 4}]
