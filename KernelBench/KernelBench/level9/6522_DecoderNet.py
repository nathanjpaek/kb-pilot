import torch
import torch.nn.functional as F
import torch.nn as nn


class DecoderNet(nn.Module):
    """
    The decoder takes an interpolated feature vector and turn it into the
    output signal. This net is intended to be very lightweight, it has only one
    hidden layer.
    """

    def __init__(self, feature_size, signal_dimension, hidden_layer_size=64):
        """
        @param feature_size dimension of an input feature vector (C)
        @param signal_dimension number of component in the output signal
                                e.g. 3 or 4 for an image, 1 for a signed
                                distance field, etc.
        @param hidden_layer_size number of neurons in the hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(feature_size, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, signal_dimension)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x * 0.5 + 0.5


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'feature_size': 4, 'signal_dimension': 4}]
