import torch
import numpy as np
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.utils.data.distributed
import torch.utils.data


def gelu(x):
    return 0.5 * x * (1 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 *
        torch.pow(x, 3))))


class StackingNNet(nn.Module):

    def __init__(self, input_size, output_size):
        super(StackingNNet, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size)

    def forward(self, x, labels=None):
        x = self.fc1(x)
        outputs = gelu(x),
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(outputs[0], labels)
            outputs = (loss,) + outputs
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
