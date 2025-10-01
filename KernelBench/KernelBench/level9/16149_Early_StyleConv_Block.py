import math
import torch
import torch.nn as nn


def quick_scale(module, name='weight'):
    ScaleW.apply(module, name)
    return module


class ScaleW:
    """
    Constructor: name - name of attribute to be scaled
    """

    def __init__(self, name):
        self.name = name

    def scale(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()
        return weight * math.sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        """
        Apply runtime scaling to specific module
        """
        hook = ScaleW(name)
        weight = getattr(module, name)
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        del module._parameters[name]
        module.register_forward_pre_hook(hook)

    def __call__(self, module, whatever):
        weight = self.scale(module)
        setattr(module, self.name, weight)


class SLinear(nn.Module):

    def __init__(self, dim_in, dim_out):
        super().__init__()
        linear = nn.Linear(dim_in, dim_out)
        linear.weight.data.normal_()
        linear.bias.data.zero_()
        self.linear = quick_scale(linear)

    def forward(self, x):
        return self.linear(x)


class SConv2d(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()
        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = quick_scale(conv)

    def forward(self, x):
        return self.conv(x)


class FC_A(nn.Module):
    """
    Learned affine transform A, this module is used to transform
    midiate vector w into a style vector
    """

    def __init__(self, dim_latent, n_channel):
        super().__init__()
        self.transform = SLinear(dim_latent, n_channel * 2)
        self.transform.linear.bias.data[:n_channel] = 1
        self.transform.linear.bias.data[n_channel:] = 0

    def forward(self, w):
        style = self.transform(w).unsqueeze(2).unsqueeze(3)
        return style


class AdaIn(nn.Module):
    """
    adaptive instance normalization
    """

    def __init__(self, n_channel):
        super().__init__()
        self.norm = nn.InstanceNorm2d(n_channel)

    def forward(self, image, style):
        factor, bias = style.chunk(2, 1)
        result = self.norm(image)
        result = result * factor + bias
        return result


class Scale_B(nn.Module):
    """
    Learned per-channel scale factor, used to scale the noise
    """

    def __init__(self, n_channel):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((1, n_channel, 1, 1)))

    def forward(self, noise):
        result = noise * self.weight
        return result


class Early_StyleConv_Block(nn.Module):
    """
    This is the very first block of generator that get the constant value as input
    """

    def __init__(self, n_channel, dim_latent, dim_input):
        super().__init__()
        self.constant = nn.Parameter(torch.randn(1, n_channel, dim_input,
            dim_input))
        self.style1 = FC_A(dim_latent, n_channel)
        self.style2 = FC_A(dim_latent, n_channel)
        self.noise1 = quick_scale(Scale_B(n_channel))
        self.noise2 = quick_scale(Scale_B(n_channel))
        self.adain = AdaIn(n_channel)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv = SConv2d(n_channel, n_channel, 3, padding=1)

    def forward(self, latent_w, noise):
        result = self.constant.repeat(noise.shape[0], 1, 1, 1)
        result = result + self.noise1(noise)
        result = self.adain(result, self.style1(latent_w))
        result = self.lrelu(result)
        result = self.conv(result)
        result = result + self.noise2(noise)
        result = self.adain(result, self.style2(latent_w))
        result = self.lrelu(result)
        return result


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_channel': 4, 'dim_latent': 4, 'dim_input': 4}]
