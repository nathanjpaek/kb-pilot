import torch
import torch.nn as nn


class Evidential_layer(nn.Module):

    def __init__(self, in_dim, num_classes):
        super(Evidential_layer, self).__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_dim, 2 * self.num_classes)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        return self.relu(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'num_classes': 4}]
