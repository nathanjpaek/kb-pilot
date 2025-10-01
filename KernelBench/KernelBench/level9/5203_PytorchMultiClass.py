import torch
import torch.nn as nn
import torch.nn.functional as F


class PytorchMultiClass(nn.Module):

    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        self.layer_1 = nn.Linear(num_features, 80)
        self.layer_2 = nn.Linear(80, 100)
        self.layer_out = nn.Linear(100, 104)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = F.dropout(F.relu(self.layer_2(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
