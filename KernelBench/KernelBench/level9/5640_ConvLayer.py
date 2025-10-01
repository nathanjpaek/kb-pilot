import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class ConvLayer(nn.Module):

    def __init__(self, input_units, output_units, filter_size,
        padding_sizes, dropout=0.2):
        super(ConvLayer, self).__init__()
        self.conv = weight_norm(nn.Conv1d(in_channels=input_units,
            out_channels=output_units, kernel_size=filter_size[0], stride=1,
            padding=padding_sizes[0]))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv, self.relu, self.dropout)
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)

    def forward(self, input_x):
        """
        Convolution Layer.
        Args:
            input_x: [batch_size, embedding_size, sequence_length]
        Returns:
            conv_out: [batch_size, sequence_length, num_filters]
            conv_avg: [batch_size, num_filters]
        """
        conv_out = self.net(input_x)
        conv_out = conv_out.permute(0, 2, 1)
        conv_avg = torch.mean(conv_out, dim=1)
        return conv_out, conv_avg


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_units': 4, 'output_units': 4, 'filter_size': [4, 4],
        'padding_sizes': [4, 4]}]
