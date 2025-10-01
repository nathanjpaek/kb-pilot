import torch
from torch import nn
import torch.utils.data


class AddSubNet(nn.Module):
    """
    Simple AddSub network in PyTorch. This network outputs the sum and
    subtraction of the inputs.
    """

    def __init__(self):
        super(AddSubNet, self).__init__()

    def forward(self, input0, input1):
        """ 
        """
        return input0 + input1, input0 - input1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
