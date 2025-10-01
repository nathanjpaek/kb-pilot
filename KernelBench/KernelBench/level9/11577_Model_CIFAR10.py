import torch
import torch.nn as nn
import torch.nn.functional as F


class Model_CIFAR10(nn.Module):

    def __init__(self):
        super(Model_CIFAR10, self).__init__()
        self.linear1 = nn.Linear(32 * 32, 50)
        self.linear2 = nn.Linear(50, 10)

    def forward(self, inputs):
        x = inputs.view(-1, 32 * 32)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)


def get_inputs():
    return [torch.rand([4, 1024])]


def get_init_inputs():
    return [[], {}]
