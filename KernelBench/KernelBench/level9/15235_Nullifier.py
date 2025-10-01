import torch
import torch.nn as nn


class Nullifier(nn.Container):

    def __init__(self):
        super(Nullifier, self).__init__()

    def forward(self, inTensor):
        outTensor = inTensor.clone()
        outTensor.fill_(0.0)
        return outTensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
