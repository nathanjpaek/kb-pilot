import torch
from torch import nn


class AddSubNet(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(AddSubNet, self).__init__()

    def forward(self, input0, input1):
        return torch.sub(input0, input1, alpha=-1), torch.sub(input0,
            input1, alpha=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
