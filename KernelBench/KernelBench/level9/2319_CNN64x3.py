import torch
import torch.nn as nn


class CNN64x3(nn.Module):

    def __init__(self, input_channels, output_channels):
        super(CNN64x3, self).__init__()
        self.conv = nn.Conv2d(in_channels=input_channels, kernel_size=3,
            out_channels=output_channels)
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(5, stride=3, padding=0)

    def forward(self, batch_data):
        output = self.conv(batch_data)
        output = self.relu(output)
        output = self.pool(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'output_channels': 4}]
