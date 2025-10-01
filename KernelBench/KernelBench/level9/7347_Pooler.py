import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *


def linear(x):
    return x


def activation(func_a):
    """Activation function wrapper
    """
    try:
        f = eval(func_a)
    except:
        f = linear
    return f


class DropoutWrapper(nn.Module):
    """
    This is a dropout wrapper which supports the fix mask dropout
    """

    def __init__(self, dropout_p=0, enable_vbp=True):
        super(DropoutWrapper, self).__init__()
        """variational dropout means fix dropout mask
        ref: https://discuss.pytorch.org/t/dropout-for-rnns/633/11
        """
        self.enable_variational_dropout = enable_vbp
        self.dropout_p = dropout_p

    def forward(self, x):
        """
            :param x: batch * len * input_size
        """
        if self.training is False or self.dropout_p == 0:
            return x
        if len(x.size()) == 3:
            mask = 1.0 / (1 - self.dropout_p) * torch.bernoulli((1 - self.
                dropout_p) * (x.data.new(x.size(0), x.size(2)).zero_() + 1))
            mask.requires_grad = False
            return mask.unsqueeze(1).expand_as(x) * x
        else:
            return F.dropout(x, p=self.dropout_p, training=self.training)


class Pooler(nn.Module):

    def __init__(self, hidden_size, dropout_p=0.1, actf='tanh'):
        super(Pooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = activation(actf)
        self.dropout = DropoutWrapper(dropout_p=dropout_p)

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        first_token_tensor = self.dropout(first_token_tensor)
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
