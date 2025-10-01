import torch
import torch.nn as nn


class GreedyCTCDecoder(nn.Module):
    """ Greedy CTC Decoder
    """

    def __init__(self, **kwargs):
        nn.Module.__init__(self)

    def forward(self, log_probs):
        with torch.no_grad():
            argmx = log_probs.argmax(dim=-1, keepdim=False).int()
            return argmx


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
