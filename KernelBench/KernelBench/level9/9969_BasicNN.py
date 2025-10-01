import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class BasicNN(nn.Module):

    def __init__(self):
        super(BasicNN, self).__init__()
        self.net = nn.Linear(28 * 28, 2)

    def forward(self, x):
        if type(x) == np.ndarray:
            x = torch.from_numpy(x)
        x = x.float()
        x = Variable(x)
        x = x.view(1, 1, 28, 28)
        x = x / 255.0
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        output = self.net(x.float())
        return F.softmax(output)


def get_inputs():
    return [torch.rand([1, 1, 28, 28])]


def get_init_inputs():
    return [[], {}]
