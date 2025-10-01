import torch
import torch.nn as nn


class CausalConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        dilation=1, groups=1, bias=True):
        self.padding = (kernel_size - 1) * dilation
        super(CausalConv1d, self).__init__(in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=self.padding,
            dilation=dilation, groups=groups, bias=bias)

    def forward(self, inputs):
        results = super(CausalConv1d, self).forward(inputs)
        padding = self.padding[0]
        if padding != 0:
            return results[:, :, :-padding]
        return results


class Inception_Temporal_Layer(nn.Module):

    def __init__(self, num_stations, In_channels, Hid_channels, Out_channels):
        super(Inception_Temporal_Layer, self).__init__()
        self.temporal_conv1 = CausalConv1d(In_channels, Hid_channels, 3,
            dilation=1, groups=1)
        self.temporal_conv2 = CausalConv1d(Hid_channels, Hid_channels, 2,
            dilation=2, groups=1)
        self.temporal_conv3 = CausalConv1d(Hid_channels, Hid_channels, 2,
            dilation=4, groups=1)
        self.conv1_1 = CausalConv1d(3 * Hid_channels, Out_channels, 1)
        self.num_stations = num_stations
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, inputs):
        output_1 = torch.cat([self.temporal_conv1(inputs[:, s_i].transpose(
            1, 2)).transpose(1, 2).unsqueeze(1) for s_i in range(self.
            num_stations)], dim=1)
        output_1 = self.act(output_1)
        output_2 = torch.cat([self.temporal_conv2(output_1[:, s_i].
            transpose(1, 2)).transpose(1, 2).unsqueeze(1) for s_i in range(
            self.num_stations)], dim=1)
        output_2 = self.act(output_2)
        output_3 = torch.cat([self.temporal_conv3(output_2[:, s_i].
            transpose(1, 2)).transpose(1, 2).unsqueeze(1) for s_i in range(
            self.num_stations)], dim=1)
        output_3 = self.act(output_3)
        outputs = torch.cat([output_1, output_2, output_3], dim=-1)
        outputs = torch.cat([self.conv1_1(outputs[:, s_i].transpose(1, 2)).
            transpose(1, 2).unsqueeze(1) for s_i in range(self.num_stations
            )], dim=1)
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_stations': 4, 'In_channels': 4, 'Hid_channels': 4,
        'Out_channels': 4}]
