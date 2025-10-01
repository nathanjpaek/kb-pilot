import torch
import torch.nn as nn


class Quant_Distribution_Loss(nn.Module):

    def __init__(self):
        super(Quant_Distribution_Loss, self).__init__()

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'
        ) ->torch.Tensor:
        m = input * target
        n = target * target
        k = m.sum() / n.sum()
        return (k - 1).abs()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
