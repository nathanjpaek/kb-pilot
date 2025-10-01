import torch
import torch.nn as nn


class ODDetector(nn.Module):

    def __init__(self, input_dim, h_size, num_classes):
        super(ODDetector, self).__init__()
        self.relu = nn.ReLU(True)
        self.fc1 = nn.Linear(input_dim, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.classifier = nn.Linear(h_size, num_classes)

    def forward(self, x, center_loss=False):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        pred = self.classifier(h)
        return pred


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'h_size': 4, 'num_classes': 4}]
