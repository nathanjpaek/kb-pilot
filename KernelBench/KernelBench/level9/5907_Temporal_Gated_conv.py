import torch
import torch.nn as nn


class Temporal_Gated_conv(nn.Module):
    """
    时序卷积模块，通过一位卷积提取时序关系
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding=0,
        stride=1):
        super(Temporal_Gated_conv, self).__init__()
        self.conv_1 = nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding, stride=
            stride)
        self.conv_2 = nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding, stride=
            stride)
        self.conv_3 = nn.Conv1d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=kernel_size, padding=padding, stride=
            stride)

    def forward(self, X):
        """

        :param X: input size of X is [batch_size,num_node,in_channel]
        :return: output size is [batch_size,num_node,out_channel]
        """
        X = X.permute(0, 2, 1)
        sig_re = torch.sigmoid(self.conv_2(X))
        GLU_result = self.conv_1(X).mul(sig_re)
        conv_x_3 = self.conv_3(X)
        temporal_result = GLU_result + conv_x_3
        temporal_result = temporal_result.permute(0, 2, 1)
        return temporal_result


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'kernel_size': 4}]
