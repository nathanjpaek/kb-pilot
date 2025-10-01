import torch


class SPPLayer(torch.nn.Module):

    def __init__(self, level):
        super(SPPLayer, self).__init__()
        self.level = level

    def forward(self, x):
        _n, _c, _h, _w = x.size()
        a = 6 + (self.level - 1) * -2
        zero_pad = torch.nn.ZeroPad2d((a, a, a, a))
        x = zero_pad(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'level': 4}]
