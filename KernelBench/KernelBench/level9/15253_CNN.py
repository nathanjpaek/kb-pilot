import torch
import torch.nn as nn
import torch._utils


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Conv2d(1, 1, 3, stride=1, padding=1)

    def forward(self, input):
        output = self.cnn(input)
        return output


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
