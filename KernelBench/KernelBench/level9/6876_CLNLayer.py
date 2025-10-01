import torch
import torch.nn as nn
import torch.nn.functional as F


class CLN(nn.Module):

    def __init__(self, in_dim, use_style_fc=False, style_dim=None,
        which_linear=nn.Linear, spectral_norm=False, eps=1e-05, **kwargs):
        super(CLN, self).__init__()
        self.in_dim = in_dim
        self.use_style_fc = use_style_fc
        self.style_dim = style_dim
        self.spectral_norm = spectral_norm
        if use_style_fc:
            self.gain = which_linear(style_dim, in_dim)
            self.bias = which_linear(style_dim, in_dim)
            if spectral_norm:
                self.gain = nn.utils.spectral_norm(self.gain)
                self.bias = nn.utils.spectral_norm(self.bias)
        else:
            self.style_dim = in_dim * 2
        self.eps = eps
        pass

    def forward(self, x, style):
        """
    Calculate class-conditional gains and biases.

    :param x: (b, c) or (b, n, c)
    :param style: (b, c)
    :return:
    """
        if self.use_style_fc:
            gain = self.gain(style) + 1.0
            bias = self.bias(style)
        else:
            style = rearrange(style, 'b (n c) -> b n c', n=2)
            gain, bias = style.unbind(dim=1)
            gain = gain + 1.0
        if x.dim() == 3:
            gain = rearrange(gain, 'b c -> b 1 c')
            bias = rearrange(bias, 'b c -> b 1 c')
        elif x.dim() == 2:
            pass
        else:
            assert 0
        out = F.layer_norm(x, normalized_shape=(self.in_dim,), weight=None,
            bias=None, eps=self.eps)
        out = out * gain + bias
        return out

    def __repr__(self):
        s = (
            f'{self.__class__.__name__}(in_dim={self.in_dim}, style_dim={self.style_dim})'
            )
        return s


class CLNLayer(nn.Module):

    def __repr__(self):
        return f'{self.__class__.__name__}({self.repr})'

    def __init__(self, in_dim, out_dim, style_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.style_dim = style_dim
        self.repr = (
            f'in_dim={in_dim}, out_dim={out_dim}, style_dim={style_dim}')
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.cln1 = CLN(in_dim=out_dim, use_style_fc=True, style_dim=style_dim)
        self.style_dim = self.cln1.style_dim
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        pass

    def forward(self, x, style):
        x = self.linear1(x)
        x = self.cln1(x, style)
        x = self.act1(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'style_dim': 4}]
