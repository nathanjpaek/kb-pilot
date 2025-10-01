import torch
import torch.nn as nn


def deconv3x3(in_planes, out_planes, stride=1, output_padding=0):
    """3x3 deconvolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=
        stride, padding=1, output_padding=output_padding, bias=False)


class DeResNetBlockGroupNorm(nn.Module):

    def __init__(self, inplanes, planes, num_groups, stride=1,
        output_padding=0, activation='relu'):
        super(DeResNetBlockGroupNorm, self).__init__()
        assert activation in ['relu', 'elu', 'leaky_relu']
        self.deconv1 = deconv3x3(inplanes, planes, stride, output_padding)
        self.gn1 = nn.GroupNorm(num_groups, planes)
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        else:
            self.activation = nn.LeakyReLU(inplace=True, negative_slope=0.1)
        self.deconv2 = deconv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(num_groups, planes)
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(nn.ConvTranspose2d(inplanes, planes,
                kernel_size=1, stride=stride, output_padding=output_padding,
                bias=False), nn.GroupNorm(num_groups, planes))
        self.downsample = downsample
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.gn1.weight, 1.0)
        nn.init.constant_(self.gn1.bias, 0.0)
        nn.init.constant_(self.gn2.weight, 1.0)
        nn.init.constant_(self.gn2.bias, 0.0)
        if self.downsample is not None:
            assert isinstance(self.downsample[1], nn.GroupNorm)
            nn.init.constant_(self.downsample[1].weight, 1.0)
            nn.init.constant_(self.downsample[1].bias, 0.0)

    def init(self, x, init_scale=1.0):
        with torch.no_grad():
            return self(x)

    def forward(self, x):
        residual = x
        out = self.deconv1(x)
        out = self.gn1(out)
        out = self.activation(out)
        out = self.deconv2(out)
        out = self.gn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4, 'num_groups': 1}]
