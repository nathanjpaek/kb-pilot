import torch
import torch.cuda
from torch import nn
import torch.distributed
import torch.utils.data
import torch.optim


def str2act(txt):
    """Translates text to neural network activation"""
    return {'sigmoid': nn.Sigmoid(), 'relu': nn.ReLU(), 'none': nn.
        Sequential(), 'lrelu': nn.LeakyReLU(0.2), 'selu': nn.SELU()}[txt.
        lower()]


class ConvGLU(nn.Module):
    """
    A convGlu operation, used by the Degli paper's model.
    """

    def __init__(self, in_ch, out_ch, kernel_size=(7, 7), padding=None,
        batchnorm=False, act='sigmoid', stride=None):
        super().__init__()
        if not padding:
            padding = kernel_size[0] // 2, kernel_size[1] // 2
        if stride is None:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=
                padding)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, padding=
                padding, stride=stride)
        self.weight = self.conv.weight
        self.bias = self.conv.bias
        if batchnorm:
            self.conv = nn.Sequential(self.conv, nn.BatchNorm2d(out_ch * 2))
        self.sigmoid = str2act(act)

    def forward(self, x):
        x = self.conv(x)
        ch = x.shape[1]
        x = x[:, :ch // 2, ...] * self.sigmoid(x[:, ch // 2:, ...])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_ch': 4, 'out_ch': 4}]
