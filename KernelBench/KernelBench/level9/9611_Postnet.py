import torch
from torch import nn


class Postnet(nn.Module):
    """Postnet is a simple linear layer for predicting the target frames given the
    RNN context during training. We don't need the Postnet for feature extraction.
    """

    def __init__(self, input_size, output_size=80):
        super(Postnet, self).__init__()
        self.layer = nn.Conv1d(in_channels=input_size, out_channels=
            output_size, kernel_size=1, stride=1)

    def forward(self, inputs):
        inputs = torch.transpose(inputs, 1, 2)
        return torch.transpose(self.layer(inputs), 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
