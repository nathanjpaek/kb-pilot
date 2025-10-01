import torch
from torch import nn
import torch.utils.data
from typing import Tuple


class LastTimeStep(nn.Module):
    """
    A class for extracting the hidden activations of the last time step following 
    the output of a PyTorch RNN module. 
    """

    def __init__(self, bidirectional=False):
        super(LastTimeStep, self).__init__()
        if bidirectional:
            self.num_driections = 2
        else:
            self.num_driections = 1

    def forward(self, x: 'Tuple'):
        last_step = x[0]
        batch_size = last_step.shape[1]
        seq_len = last_step.shape[0]
        last_step = last_step.view(seq_len, batch_size, self.num_driections, -1
            )
        last_step = torch.mean(last_step, 2)
        last_step = last_step[0]
        return last_step.reshape(batch_size, -1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
