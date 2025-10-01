import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable


def seq_dropout(x, p=0, training=False):
    """
    x: batch * len * input_size
    """
    if training is False or p == 0:
        return x
    dropout_mask = Variable(1.0 / (1 - p) * torch.bernoulli((1 - p) * (x.
        data.new(x.size(0), x.size(2)).zero_() + 1)), requires_grad=False)
    return dropout_mask.unsqueeze(1).expand_as(x) * x


def dropout(x, p=0, training=False):
    """
    x: (batch * len * input_size) or (any other shape)
    """
    if len(x.size()) == 3:
        return seq_dropout(x, p=p, training=training)
    else:
        return F.dropout(x, p=p, training=training)


class MLPFunc(nn.Module):
    """
    A multi-layer perceptron function for x: o = v'tanh(Wx+b).
    """

    def __init__(self, input_size, hidden_size, num_class):
        super(MLPFunc, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.linear_final = nn.Linear(hidden_size, num_class, bias=False)

    def forward(self, x):
        """
        x = batch * input_size
        """
        x = dropout(x, p=0.2, training=self.training)
        h = F.tanh(self.linear(x))
        h = dropout(h, p=0.2, training=self.training)
        o = self.linear_final(h)
        return o


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'num_class': 4}]
