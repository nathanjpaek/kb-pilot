import torch
import torch.cuda
import torch.nn
import torch.utils.data
import torch.fx
import torch.utils.tensorboard._pytorch_graph


class Concat(torch.nn.Module):
    """ Concat module for a functional concat"""

    def __init__(self, axis: 'int'=0):
        super(Concat, self).__init__()
        self.axis = axis

    def forward(self, x, y):
        """
        Forward-pass routine for divide op
        """
        return torch.cat((x, y), self.axis)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
