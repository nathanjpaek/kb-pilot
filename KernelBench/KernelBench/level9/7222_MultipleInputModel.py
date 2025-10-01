import torch
from torch import nn as nn
from torch import optim as optim


class TemplateModel(nn.Module):

    def __init__(self, mix_data=False):
        """ Base model for testing. The setting ``mix_data=True`` simulates a wrong implementation. """
        super().__init__()
        self.mix_data = mix_data
        self.linear = nn.Linear(10, 5)
        self.input_array = torch.rand(10, 5, 2)

    def forward(self, *args, **kwargs):
        return self.forward__standard(*args, **kwargs)

    def forward__standard(self, x):
        if self.mix_data:
            x = x.view(10, -1).permute(1, 0).view(-1, 10)
        else:
            x = x.view(-1, 10)
        return self.linear(x)


class MultipleInputModel(TemplateModel):
    """ Base model for testing verification when forward accepts multiple arguments. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_array = torch.rand(10, 5, 2), torch.rand(10, 5, 2)

    def forward(self, x, y, some_kwarg=True):
        out = super().forward(x) + super().forward(y)
        return out


def get_inputs():
    return [torch.rand([4, 10]), torch.rand([4, 10])]


def get_init_inputs():
    return [[], {}]
