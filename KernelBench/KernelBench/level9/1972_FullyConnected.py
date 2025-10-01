import torch
import torch.utils.data
import torch.nn as nn


def _init_weights(layer):
    """
    Init weights of the layer
    :param layer:
    :return:
    """
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


class FullyConnected(nn.Module):

    def __init__(self, in_features, out_features, activation_fn=nn.
        functional.relu):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        _init_weights(self.fc)
        self.activation = activation_fn

    def forward(self, input):
        out = self.fc(input)
        if self.activation is not None:
            out = self.activation(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
