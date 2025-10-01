import torch
import torch.nn as nn
import torch.nn.functional as F


class L2Norm(nn.Module):

    def forward(self, x):
        if len(x.size()) > 1:
            return x / x.norm(p=2, dim=1, keepdim=True)
        else:
            return x / x.norm(p=2)


class NonLinearModel(nn.Module):

    def __init__(self, inputs, outputs, hiddens=32):
        super(NonLinearModel, self).__init__()
        self.l1 = nn.Linear(inputs, hiddens)
        self.l2 = nn.Linear(hiddens, outputs)
        self.norm = L2Norm()

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        x = self.norm(x)
        return x


class ComposeModel(nn.Module):

    def __init__(self, inputs, outputs, hiddens=16):
        super(ComposeModel, self).__init__()
        self.encoder = NonLinearModel(inputs, hiddens)
        self.output = NonLinearModel(hiddens, outputs)

    def forward(self, x):
        x = self.encoder(x)
        return self.output(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inputs': 4, 'outputs': 4}]
