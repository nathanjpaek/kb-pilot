import torch
import torch.optim


class Snake(torch.nn.Module):

    def __init__(self, alpha: 'float'=1.0):
        super(Snake, self).__init__()
        self.alpha = alpha
        self.one_over_alpha = 1.0 / alpha

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        s = torch.sin(self.alpha * x)
        return x + self.one_over_alpha * s ** 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
