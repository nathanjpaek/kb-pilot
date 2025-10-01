import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal


def ZeroInitializer(param):
    shape = param.size()
    init = np.zeros(shape).astype(np.float32)
    param.data.set_(torch.from_numpy(init))


def Linear(initializer=kaiming_normal, bias_initializer=ZeroInitializer):


    class CustomLinear(nn.Linear):

        def reset_parameters(self):
            initializer(self.weight)
            if self.bias is not None:
                bias_initializer(self.bias)
    return CustomLinear


class LayerNormalization(nn.Module):

    def __init__(self, hidden_size, eps=1e-05):
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.a2 = nn.Parameter(torch.ones(1, hidden_size), requires_grad=True)
        self.b2 = nn.Parameter(torch.zeros(1, hidden_size), requires_grad=True)

    def forward(self, z):
        mu = torch.mean(z)
        sigma = torch.std(z)
        ln_out = (z - mu) / (sigma + self.eps)
        ln_out = ln_out * self.a2 + self.b2
        return ln_out


class MLP(nn.Module):

    def __init__(self, mlp_input_dim, mlp_dim, num_classes, num_mlp_layers,
        mlp_ln, classifier_dropout_rate=0.0):
        super(MLP, self).__init__()
        self.num_mlp_layers = num_mlp_layers
        self.mlp_ln = mlp_ln
        self.classifier_dropout_rate = classifier_dropout_rate
        features_dim = mlp_input_dim
        if mlp_ln:
            self.ln_inp = LayerNormalization(mlp_input_dim)
        for i in range(num_mlp_layers):
            setattr(self, 'l{}'.format(i), Linear()(features_dim, mlp_dim))
            if mlp_ln:
                setattr(self, 'ln{}'.format(i), LayerNormalization(mlp_dim))
            features_dim = mlp_dim
        setattr(self, 'l{}'.format(num_mlp_layers), Linear()(features_dim,
            num_classes))

    def forward(self, h):
        if self.mlp_ln:
            h = self.ln_inp(h)
        h = F.dropout(h, self.classifier_dropout_rate, training=self.training)
        for i in range(self.num_mlp_layers):
            layer = getattr(self, 'l{}'.format(i))
            h = layer(h)
            h = F.relu(h)
            if self.mlp_ln:
                ln = getattr(self, 'ln{}'.format(i))
                h = ln(h)
            h = F.dropout(h, self.classifier_dropout_rate, training=self.
                training)
        layer = getattr(self, 'l{}'.format(self.num_mlp_layers))
        y = layer(h)
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'mlp_input_dim': 4, 'mlp_dim': 4, 'num_classes': 4,
        'num_mlp_layers': 1, 'mlp_ln': 4}]
