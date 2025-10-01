import torch
import torch.nn.functional as F


class HardSigmoid(torch.nn.Module):
    """
    Pytorch implementation of the hard sigmoid activation function
    """

    def __init__(self):
        super(HardSigmoid, self).__init__()

    def forward(self, input):
        x = 0.2 * input + 0.5
        x = torch.clamp(x, 0, 1)
        x = F.threshold(-x, -1, -1)
        x = F.threshold(-x, 0, 0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
