import torch
import torch.nn as nn


class FlowHead(nn.Module):

    def __init__(self, input_dim=128, hidden_dim=256, output_dim=2):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, output_dim, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


def get_inputs():
    return [torch.rand([4, 128, 64, 64])]


def get_init_inputs():
    return [[], {}]
