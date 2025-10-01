import torch


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, padding=0):
    """3x3 convolution with padding"""
    return torch.nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=padding, groups=groups, bias=False, dilation=dilation)


class BasicBlock(torch.nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=
        1, base_width=64, dilation=1, activation_fct='relu', norm_layer=None):
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=
            stride, padding=(1, 0))
        self.activation = torch.nn.ReLU(inplace=True
            ) if activation_fct == 'relu' else torch.nn.Tanh()
        self.conv2 = conv3x3(in_planes=planes, out_planes=planes, padding=(
            1, 0))
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = torch.nn.functional.pad(input=x, pad=(1, 1, 0, 0), mode=
            'circular')
        out = self.conv1(out)
        out = self.activation(out)
        out = torch.nn.functional.pad(input=out, pad=(1, 1, 0, 0), mode=
            'circular')
        out = self.conv2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
