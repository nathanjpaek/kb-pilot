import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvFC(nn.Module):

    def __init__(self, conv_in_channels, conv_out_channels, input_size,
        hidden_size, output_size, kernel_size=3):
        super(ConvFC, self).__init__()
        self.conv_out_channels = conv_out_channels
        self.layer1 = nn.Conv2d(conv_in_channels, conv_out_channels,
            kernel_size=kernel_size)
        self.conv_result_size = input_size - kernel_size + 1
        self.fc_size = self.conv_result_size ** 2 * self.conv_out_channels
        self.layer2 = nn.Linear(self.fc_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        assert len(x.shape) >= 3
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        conv_output = F.leaky_relu(self.layer1(x))
        output = conv_output.reshape(-1, self.fc_size)
        output = F.leaky_relu(self.layer2(output))
        output = self.layer3(output)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'conv_in_channels': 4, 'conv_out_channels': 4,
        'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
