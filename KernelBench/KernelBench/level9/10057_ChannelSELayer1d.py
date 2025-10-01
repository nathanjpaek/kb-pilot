import torch
import torch.nn as nn


class ChannelSELayer1d(nn.Module):

    def __init__(self, num_channels, reduction_ratio=4):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer1d, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.activ_1 = nn.ReLU()
        self.activ_2 = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H)
        :return: output tensor
        """
        batch_size, num_channels, _H = input_tensor.size()
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(
            dim=2)
        fc_out_1 = self.activ_1(self.fc1(squeeze_tensor))
        fc_out_2 = self.activ_2(self.fc2(fc_out_1))
        return input_tensor * fc_out_2.view(batch_size, num_channels, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'num_channels': 4}]
