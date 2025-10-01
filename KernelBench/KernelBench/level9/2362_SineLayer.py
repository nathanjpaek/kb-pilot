import torch
import numpy as np
import torch.nn as nn


class SineLayer(nn.Module):

    def __init__(self, in_features: 'int', out_features: 'int', omega_0:
        'float'=30, is_first: 'bool'=False) ->None:
        """Sine activation function layer with omega_0 scaling.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            omega_0 (float, optional): Scaling factor of the Sine function. Defaults to 30.
            is_first (bool, optional): Defaults to False.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self) ->None:
        """Initialization of the weigths."""
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self
                    .in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) /
                    self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input: 'torch.Tensor') ->torch.Tensor:
        """Forward pass through the layer.

        Args:
            input (torch.Tensor): Input tensor of shape (n_samples, n_inputs).

        Returns:
            torch.Tensor: Prediction of shape (n_samples, n_outputs)
        """
        return torch.sin(self.omega_0 * self.linear(input))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
