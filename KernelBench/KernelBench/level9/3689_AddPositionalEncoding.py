import torch
import torch.nn as nn
import torch.onnx


class AddPositionalEncoding(nn.Module):

    def __init__(self, hidden_size, max_sequence_length):
        super(AddPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = nn.Parameter(torch.empty(
            max_sequence_length, hidden_size))
        nn.init.normal_(self.positional_encoding)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_encoding[:seq_len]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'max_sequence_length': 4}]
