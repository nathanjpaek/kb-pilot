import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):

    def __init__(self, input_shape, output_shape):
        super(Decoder, self).__init__()
        self.input_shape = input_shape
        self.decoder_out_shape = output_shape
        self.linear_one = nn.Linear(self.input_shape, 100)
        self.linear_two = nn.Linear(100, 200)
        self.linear_three = nn.Linear(200, 400)
        self.linear_four = nn.Linear(400, self.decoder_out_shape)

    def forward(self, x):
        x = F.relu(self.linear_one(x))
        x = F.relu(self.linear_two(x))
        x = F.relu(self.linear_three(x))
        x = self.linear_four(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4, 'output_shape': 4}]
