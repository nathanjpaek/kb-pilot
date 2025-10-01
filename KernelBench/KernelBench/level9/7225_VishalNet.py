import torch
import torch.nn as nn


class VishalNet(nn.Module):

    def __init__(self):
        super(VishalNet, self).__init__()
        self.cnn1 = nn.Conv1d(1, 60, 81, 1, 40)
        self.cnn2 = nn.Conv1d(60, 1, 301, 1, 150)

    def forward(self, input):
        out1 = nn.functional.relu(self.cnn1(input))
        out2 = self.cnn2(out1)
        return out2


def get_inputs():
    return [torch.rand([4, 1, 64])]


def get_init_inputs():
    return [[], {}]
