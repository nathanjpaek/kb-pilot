import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class TinyDiscriminator(nn.Module):

    def __init__(self, n_features, n_classes=1, d_hidden=128):
        super(TinyDiscriminator, self).__init__()
        self.n_features = n_features
        self.n_classes = n_classes
        self.d_hidden = d_hidden
        self.l1 = nn.Linear(n_features, d_hidden)
        self.l2 = nn.Linear(d_hidden, 1)
        if n_classes > 1:
            self.linear_y = nn.Embedding(n_classes, d_hidden)

    def forward(self, inputs, y=None):
        output = self.l1(inputs)
        features = F.leaky_relu(output, 0.1, inplace=True)
        d = self.l2(features)
        if y is not None:
            w_y = self.linear_y(y)
            d = d + (features * w_y).sum(1, keepdim=True)
        return d


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
