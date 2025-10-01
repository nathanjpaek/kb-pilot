import torch
import torch.nn.functional as F
import torch.utils.data.distributed
import torch
import torch.nn as nn


class SLP(nn.Module):

    def __init__(self, input_size, logits):
        super(SLP, self).__init__()
        self._input_size = input_size
        self.fc = nn.Linear(input_size, logits)

    def forward(self, x):
        x = x.view(-1, self._input_size)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'logits': 4}]
