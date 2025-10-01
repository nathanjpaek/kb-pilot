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


class SkipConnection(nn.Module):
    """
    Connects the two given inputs with concatenation
    :param in1: earlier input tensor of shape N x d1 x m x m
    :param in2: later input tensor of shape N x d2 x m x m
    :param in_features: d1+d2
    :param out_features: output num of features
    :return: Tensor of shape N x output_depth x m x m
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=1,
            padding=0, bias=True)
        _init_weights(self.conv)

    def forward(self, in1, in2):
        out = torch.cat((in1, in2), dim=1)
        out = self.conv(out)
        return out


def get_inputs():
    return [torch.rand([4, 1, 4, 4]), torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
