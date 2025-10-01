import torch
import torch.utils.data
from torch import nn
import torch
import torch.nn.parallel
import torch.optim


def Binarize(tensor, quant_mode='det'):
    if quant_mode == 'det':
        tensor = tensor.sign()
        zero = torch.zeros_like(tensor)
        one = torch.ones_like(tensor)
        zero - one
        tensor = torch.where(tensor == 0, one, tensor)
        return tensor
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)
            ).clamp_(0, 1).round().mul_(2).add_(-1)


def Binaryconv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return BinarizeConv2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=1, bias=False)


class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)

    def forward(self, input):
        input.data = Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = Binarize(self.weight.org)
        out = nn.functional.conv2d(input, self.weight, None, self.stride,
            self.padding, self.dilation, self.groups)
        if self.bias is not None:
            self.bias.org = self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
        do_bntan=True):
        super(BasicBlock, self).__init__()
        self.conv1 = Binaryconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.tanh1 = nn.Hardtanh(inplace=True)
        self.conv2 = Binaryconv3x3(planes, planes)
        self.tanh2 = nn.Hardtanh(inplace=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.do_bntan = do_bntan
        self.stride = stride

    def forward(self, x):
        residual = x.clone()
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.tanh1(out)
        out = self.conv2(out)
        if self.downsample is not None:
            if residual.data.max() > 1:
                pdb.set_trace()
            residual = self.downsample(residual)
        out += residual
        if self.do_bntan:
            out = self.bn2(out)
            out = self.tanh2(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
