import torch
import torch.utils.data


class CausalPad(torch.nn.Module):

    def __init__(self):
        super(CausalPad, self).__init__()

    def forward(self, input):
        return torch.nn.functional.pad(input, (0, 0, 1, 0))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
