import torch
from torch.optim import *
from torch import nn


class AE_Net(nn.Module):
    """docstring for AE_Net."""

    def __init__(self, input_shape):
        super(AE_Net, self).__init__()
        self.encoder_hidden_layer = nn.Linear(in_features=input_shape,
            out_features=128)
        self.encoder_output_layer = nn.Linear(in_features=128, out_features=64)
        self.decoder_hidden_layer = nn.Linear(in_features=64, out_features=128)
        self.decoder_output_layer = nn.Linear(in_features=128, out_features
            =input_shape)

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        state_logit = self.encoder_output_layer(activation)
        code = torch.relu(state_logit)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = activation
        return reconstructed, state_logit


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4}]
