import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class HUBHardsigmoid(torch.nn.Module):
    """
    This is a hub scaled addition (x+1)/2.
    """

    def __init__(self, scale=3):
        super(HUBHardsigmoid, self).__init__()
        self.scale = scale

    def forward(self, x) ->str:
        return torch.nn.Hardsigmoid()(x * self.scale)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
