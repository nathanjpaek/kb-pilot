import torch
import torch as t
import torch.utils.data


class RoundPass(t.nn.Module):

    def forward(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
