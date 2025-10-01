import torch
import torch as t
import torch.utils.data


class GradScale(t.nn.Module):

    def forward(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad


class RoundPass(t.nn.Module):

    def forward(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad


class LsqQuan(t.nn.Module):

    def __init__(self, bit, all_positive=False, symmetric=False,
        per_channel=True):
        super(LsqQuan, self).__init__()
        self.s = t.nn.Parameter(t.zeros(1))
        if all_positive:
            self.thd_neg = 0
            self.thd_pos = 2 ** bit - 1
        elif symmetric:
            self.thd_neg = -2 ** (bit - 1) + 1
            self.thd_pos = 2 ** (bit - 1) - 1
        else:
            self.thd_neg = -2 ** (bit - 1)
            self.thd_pos = 2 ** (bit - 1) - 1
        self.grad_scale = GradScale()
        self.round_pass = RoundPass()

    def forward(self, x):
        s_grad_scale = 1.0 / (self.thd_pos * x.numel()) ** 0.5
        s_scale = self.grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = t.clamp(x, self.thd_neg, self.thd_pos)
        x = self.round_pass(x)
        x = x * s_scale
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'bit': 4}]
