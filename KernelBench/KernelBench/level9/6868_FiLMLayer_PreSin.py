import torch
import numpy as np
import torch.nn as nn


class FiLMLayer_PreSin(nn.Module):

    def __init__(self, in_dim, out_dim, style_dim, use_style_fc=True,
        which_linear=nn.Linear, **kwargs):
        super(FiLMLayer_PreSin, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.use_style_fc = use_style_fc
        self.linear = which_linear(in_dim, out_dim)
        nn.init.uniform_(self.linear.weight, -np.sqrt(9 / in_dim), np.sqrt(
            9 / in_dim))
        if use_style_fc:
            self.gain_fc = which_linear(style_dim, out_dim)
            self.bias_fc = which_linear(style_dim, out_dim)
            self.gain_fc.weight.data.mul_(0.25)
            self.gain_fc.bias.data.fill_(1)
            self.bias_fc.weight.data.mul_(0.25)
        else:
            self.style_dim = out_dim * 2
        pass

    def forward(self, x, style):
        """

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        if self.use_style_fc:
            gain = self.gain_fc(style)
            bias = self.bias_fc(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        x = self.linear(x)
        x = torch.sin(x)
        out = gain * x + bias
        return out

    def __repr__(self):
        s = (
            f'{self.__class__.__name__}(in_dim={self.in_dim}, out_dim={self.out_dim}, style_dim={self.style_dim}, use_style_fc={self.use_style_fc}, )'
            )
        return s


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'style_dim': 4}]
