import torch
import torch as t
import torch.nn as nn


class SinglePITF_Loss(nn.Module):
    """
    定义PITF的loss function
    """

    def __init__(self):
        super(SinglePITF_Loss, self).__init__()
        None

    def forward(self, r):
        return t.sum(-t.log(t.sigmoid(r)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
