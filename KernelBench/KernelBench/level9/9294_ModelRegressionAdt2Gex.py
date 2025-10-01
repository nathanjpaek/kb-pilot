import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn


class Swish(torch.autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish_module(nn.Module):

    def forward(self, x):
        return Swish.apply(x)


class ModelRegressionAdt2Gex(nn.Module):

    def __init__(self, dim_mod1, dim_mod2):
        super(ModelRegressionAdt2Gex, self).__init__()
        self.input_ = nn.Linear(dim_mod1, 512)
        self.dropout1 = nn.Dropout(p=0.0)
        self.swish = Swish_module()
        self.fc = nn.Linear(512, 512)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, dim_mod2)

    def forward(self, x):
        x = F.gelu(self.input_(x))
        x = F.gelu(self.fc(x))
        x = F.gelu(self.fc1(x))
        x = F.gelu(self.fc2(x))
        x = F.gelu(self.output(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_mod1': 4, 'dim_mod2': 4}]
