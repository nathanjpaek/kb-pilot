import torch
from torch import nn


class Attention_layer(nn.Module):

    def __init__(self, sequence_length):
        super(Attention_layer, self).__init__()
        self.input_size = sequence_length
        self.output_size = sequence_length
        self.dense = nn.Linear(sequence_length, sequence_length)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_tensor):
        y = self.softmax(self.dense(input_tensor.permute(0, 2, 1)))
        y = y.permute(0, 2, 1)
        y = input_tensor * y
        return y


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'sequence_length': 4}]
