import torch
import torch.utils.data


class Envelope(torch.nn.Module):

    def __init__(self, exponent):
        super(Envelope, self).__init__()
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p)
        x_pow_p1 = x_pow_p0 * x
        return 1.0 / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p1 * x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'exponent': 4}]
