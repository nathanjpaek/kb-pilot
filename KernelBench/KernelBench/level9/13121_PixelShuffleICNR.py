import torch
from torch import nn


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


class PixelShuffleICNR(nn.Module):

    def __init__(self, in_planes, out_planes, scale=2):
        super().__init__()
        self.conv = conv1x1(in_planes, out_planes)
        self.shuffle = nn.PixelShuffle(scale)
        kernel = self.ICNR(self.conv.weight, upscale_factor=scale)
        self.conv.weight.data.copy_(kernel)

    @staticmethod
    def ICNR(tensor, upscale_factor=2, inizializer=nn.init.kaiming_normal_):
        """Fills the input Tensor or Variable with values according to the method
        described in "Checkerboard artifact free sub-pixel convolution" https://arxiv.org/abs/1707.02937
        Andrew Aitken et al. (2017), this inizialization should be used in the
        last convolutional layer before a PixelShuffle operation
        :param tensor: an n-dimensional torch.Tensor or autograd.Variable
        :param upscale_factor: factor to increase spatial resolution by
        :param inizializer: inizializer to be used for sub_kernel inizialization
        """
        new_shape = [int(tensor.shape[0] / upscale_factor ** 2)] + list(tensor
            .shape[1:])
        sub_kernel = torch.zeros(new_shape)
        sub_kernel = inizializer(sub_kernel)
        sub_kernel = sub_kernel.transpose(0, 1)
        sub_kernel = sub_kernel.contiguous().view(sub_kernel.shape[0],
            sub_kernel.shape[1], -1)
        kernel = sub_kernel.repeat(1, 1, upscale_factor ** 2)
        transposed_shape = [tensor.shape[1]] + [tensor.shape[0]] + list(tensor
            .shape[2:])
        kernel = kernel.contiguous().view(transposed_shape)
        kernel = kernel.transpose(0, 1)
        return kernel

    def forward(self, x):
        return self.shuffle(self.conv(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4}]
