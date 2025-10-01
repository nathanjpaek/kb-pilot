import torch
import torch as t
import torch.nn as nn


class PITF_Loss(nn.Module):
    """
    定义PITF的loss function
    """

    def __init__(self):
        super(PITF_Loss, self).__init__()
        None

    def forward(self, r_p, r_ne):
        return -t.log(t.sigmoid(r_p - r_ne))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
