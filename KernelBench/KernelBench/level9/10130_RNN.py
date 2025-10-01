import torch
import torch.nn as nn


class Norm(nn.Module):
    """
    Re-usable class for either batch-norm or layer-norm (by swapping dim)
    """

    def __init__(self, n_hidden, eps=1e-08, dim=0):
        super(Norm, self).__init__()
        self.eps = eps
        self.n_hidden = n_hidden
        self.a = nn.Parameter(torch.ones(1, n_hidden), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1, n_hidden), requires_grad=True)
        self.dim = dim

    def forward(self, x):
        mean_x = torch.mean(x, dim=self.dim).expand_as(x)
        std_x = torch.std(x, dim=self.dim).expand_as(x)
        out = (x - mean_x) / (std_x + self.eps)
        out = out * self.a.expand_as(x) + self.b.expand_as(x)
        return out


class LayerNorm(Norm):

    def __init__(self, n_hidden, eps=1e-08):
        super(LayerNorm, self).__init__(n_hidden, eps, dim=1)


class RNN(nn.Module):

    def __init__(self, n_in, n_hidden, n_out, layer_norm=False):
        super(RNN, self).__init__()
        self.n_hidden = n_hidden
        self.i2h = nn.Linear(n_in + n_hidden, n_hidden)
        self.dropout = nn.Dropout()
        self.h2o = nn.Linear(n_hidden, n_out)
        self.softmax = nn.LogSoftmax()
        self.layer_norm = layer_norm
        self.aux_loss = 0
        if layer_norm:
            self.normh = LayerNorm(n_hidden)
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Sigmoid()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        hidden = self.activation(hidden)
        if self.layer_norm:
            hidden = self.normh(hidden)
        output = self.h2o(self.dropout(hidden))
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return nn.Parameter(torch.zeros(1, self.n_hidden), requires_grad=True
            ).repeat(batch_size, 1)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_hidden': 4, 'n_out': 4}]
