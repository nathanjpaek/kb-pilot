import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.cuda


class ConformerConvBlock(nn.Module):

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        super(ConformerConvBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise_conv1 = nn.Conv1d(channels, 2 * channels,
            kernel_size=1, stride=1, padding=0, bias=bias)
        self.depthwise_conv = nn.Conv1d(channels, channels, kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=channels, bias
            =bias)
        self.pointwise_conv2 = nn.Conv1d(channels, channels, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.pointwise_conv1.weight, nonlinearity=
            'relu')
        nn.init.kaiming_normal_(self.depthwise_conv.weight, nonlinearity='relu'
            )
        nn.init.kaiming_normal_(self.pointwise_conv2.weight, nonlinearity=
            'relu')
        nn.init.constant_(self.pointwise_conv1.bias, 0)
        nn.init.constant_(self.pointwise_conv2.bias, 0)
        nn.init.constant_(self.depthwise_conv.bias, 0)

    def forward(self, x):
        """
        :param x: [seq_len x bsz x hidden_size]
        :return:
        """
        x = x.transpose(0, 1).transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2).transpose(0, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'kernel_size': 1}]
