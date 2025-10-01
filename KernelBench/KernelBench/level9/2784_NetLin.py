import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class NetLin(nn.Module):

    def __init__(self):
        super(NetLin, self).__init__()
        self.liner1 = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        output = self.liner1(x)
        output = F.log_softmax(input=output, dim=1)
        return output


def get_inputs():
    return [torch.rand([4, 784])]


def get_init_inputs():
    return [[], {}]
