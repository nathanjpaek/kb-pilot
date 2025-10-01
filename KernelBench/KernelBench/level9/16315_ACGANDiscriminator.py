import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
from torchvision import utils


def global_pooling(input, pooling='mean'):
    if pooling == 'mean':
        return input.mean(3).mean(2)
    elif pooling == 'sum':
        return input.sum(3).sum(2)
    else:
        raise NotImplementedError()


class CustomConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
        padding=None, bias=True, spectral_norm=False, residual_init=True):
        super(CustomConv2d, self).__init__()
        self.residual_init = residual_init
        if padding is None:
            padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias)
        if spectral_norm:
            self.conv = utils.spectral_norm(self.conv)

    def forward(self, input):
        return self.conv(input)


class CustomLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, spectral_norm=
        False):
        super(CustomLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if spectral_norm:
            self.linear = utils.spectral_norm(self.linear)

    def forward(self, input):
        return self.linear(input)


class ConvMeanPool(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        spectral_norm=False, residual_init=True):
        super(ConvMeanPool, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size,
            bias=bias, spectral_norm=spectral_norm, residual_init=residual_init
            )

    def forward(self, input):
        output = input
        output = self.conv(output)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output
            [:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        return output


class MeanPoolConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, bias=True,
        spectral_norm=False, residual_init=True):
        super(MeanPoolConv, self).__init__()
        self.conv = CustomConv2d(in_channels, out_channels, kernel_size,
            bias=bias, spectral_norm=spectral_norm, residual_init=residual_init
            )

    def forward(self, input):
        output = input
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output
            [:, :, ::2, 1::2] + output[:, :, 1::2, 1::2]) / 4
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, resample=
        None, spectral_norm=False):
        super(ResidualBlock, self).__init__()
        if in_channels != out_channels or resample is not None:
            self.learnable_shortcut = True
        else:
            self.learnable_shortcut = False
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.conv_shortcut = ConvMeanPool(in_channels, out_channels,
                kernel_size=1, spectral_norm=spectral_norm, residual_init=False
                )
            self.conv1 = CustomConv2d(in_channels, in_channels, kernel_size
                =kernel_size, spectral_norm=spectral_norm)
            self.conv2 = ConvMeanPool(in_channels, out_channels,
                kernel_size=kernel_size, spectral_norm=spectral_norm)
        elif resample is None:
            if self.learnable_shortcut:
                self.conv_shortcut = CustomConv2d(in_channels, out_channels,
                    kernel_size=1, spectral_norm=spectral_norm,
                    residual_init=False)
            self.conv1 = CustomConv2d(in_channels, out_channels,
                kernel_size=kernel_size, spectral_norm=spectral_norm)
            self.conv2 = CustomConv2d(out_channels, out_channels,
                kernel_size=kernel_size, spectral_norm=spectral_norm)
        else:
            raise NotImplementedError()

    def forward(self, input):
        if self.learnable_shortcut:
            shortcut = self.conv_shortcut(input)
        else:
            shortcut = input
        output = input
        output = self.relu1(output)
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


class OptimizedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
        spectral_norm=False):
        super(OptimizedResidualBlock, self).__init__()
        self.conv1 = CustomConv2d(in_channels, out_channels, kernel_size=
            kernel_size, spectral_norm=spectral_norm)
        self.conv2 = ConvMeanPool(out_channels, out_channels, kernel_size=
            kernel_size, spectral_norm=spectral_norm)
        self.conv_shortcut = MeanPoolConv(in_channels, out_channels,
            kernel_size=1, spectral_norm=spectral_norm, residual_init=False)
        self.relu2 = nn.ReLU()

    def forward(self, input):
        shortcut = self.conv_shortcut(input)
        output = input
        output = self.conv1(output)
        output = self.relu2(output)
        output = self.conv2(output)
        return shortcut + output


class ACGANDiscriminator(nn.Module):

    def __init__(self, num_classes=10, channels=128, dropout=False,
        spectral_norm=False, pooling='mean'):
        super(ACGANDiscriminator, self).__init__()
        self.num_classes = num_classes
        self.channels = channels
        self.dropout = dropout
        self.spectral_norm = spectral_norm
        self.pooling = pooling
        self.block1 = OptimizedResidualBlock(3, channels, 3, spectral_norm=
            spectral_norm)
        self.block2 = ResidualBlock(channels, channels, 3, resample='down',
            spectral_norm=spectral_norm)
        self.block3 = ResidualBlock(channels, channels, 3, resample=None,
            spectral_norm=spectral_norm)
        self.block4 = ResidualBlock(channels, channels, 3, resample=None,
            spectral_norm=spectral_norm)
        self.relu5 = nn.ReLU()
        self.linear5dis = CustomLinear(channels, 1, spectral_norm=spectral_norm
            )
        self.linear5cls = CustomLinear(channels, num_classes)

    def forward(self, input, dropout=None):
        if dropout is None:
            dropout = self.dropout
        output = input
        output = self.block1(output)
        output = self.block2(output)
        output = F.dropout(output, p=0.2, training=dropout)
        output = self.block3(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.block4(output)
        output = F.dropout(output, p=0.5, training=dropout)
        output = self.relu5(output)
        out_feat = global_pooling(output, 'mean')
        output = global_pooling(output, self.pooling)
        out_dis = self.linear5dis(output)
        out_cls = self.linear5cls(out_feat)
        return out_dis.squeeze(), out_cls.squeeze(), out_feat


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
