import torch
import torch.nn as nn
import torch.distributions
import torch.utils.data


class OELoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return -torch.log_softmax(logits, 1).mean(1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
