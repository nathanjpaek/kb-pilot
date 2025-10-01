import torch
from typing import Iterable
from torch.distributions import Normal
from torch import nn as nn


def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()


class Encoder_mse(nn.Module):
    """Encodes data of ``n_input`` dimensions into a latent space of ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.

    :param n_input: The dimensionality of the input (data space)
    :param n_output: The dimensionality of the output (latent space)
    :param n_cat_list: A list containing the number of categories
                       for each category of interest. Each category will be
                       included using a one-hot encoding
    :param n_layers: The number of fully-connected hidden layers
    :param n_hidden: The number of nodes per hidden layer
    :dropout_rate: Dropout rate to apply to each of the hidden layers
    """

    def __init__(self, n_input: 'int', n_output: 'int', n_cat_list:
        'Iterable[int]'=None, n_layers: 'int'=1, n_hidden: 'int'=128,
        dropout_rate: 'float'=0.1):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: 'torch.Tensor', *cat_list: int):
        """The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\\\( q_m \\\\) and variance \\\\( q_v \\\\) (clamped to \\\\( [-5, 5] \\\\))
         #. Samples a new value from an i.i.d. multivariate normal \\\\( \\\\sim N(q_m, \\\\mathbf{I}q_v) \\\\)

        :param x: tensor with shape (n_input,)
        :param cat_list: list of category membership(s) for this sample
        :return: tensors of shape ``(n_latent,)`` for mean and var, and sample
        :rtype: 3-tuple of :py:class:`torch.Tensor`
        """
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 0.0001
        latent = reparameterize_gaussian(q_m, q_v)
        return q_m, q_v, latent


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_input': 4, 'n_output': 4}]
