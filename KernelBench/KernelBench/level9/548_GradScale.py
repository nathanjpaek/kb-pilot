import torch
import torch as t
import torch.utils.data


class GradScale(t.nn.Module):

    def forward(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
