import torch
import torch.sparse
import torch.nn as nn


class MultVAE_encoder(nn.Module):

    def __init__(self, item_dim: 'int', hidden_dim=600, latent_dim=200,
        n_hidden_layers=1, dropout=0.5, nonlinearity=nn.Tanh):
        super(MultVAE_encoder, self).__init__()
        self.item_dim = item_dim
        self.latent_dim = latent_dim
        self.nonlinearity = nn.Tanh()
        self.layers = nn.Sequential()
        self.layers.add_module('input_dropout', nn.Dropout(dropout))
        self.layers.add_module('linear_enc_1', nn.Linear(in_features=
            item_dim, out_features=hidden_dim))
        self.layers.add_module('Tanh_enc_1', self.nonlinearity)
        if n_hidden_layers > 0:
            for i in range(n_hidden_layers):
                self.layers.add_module('hidden_enc_{}'.format(i + 1), nn.
                    Linear(in_features=hidden_dim, out_features=hidden_dim))
                self.layers.add_module('Tanh_enc_{}'.format(i + 2), self.
                    nonlinearity)
        self.mu = nn.Linear(in_features=hidden_dim, out_features=latent_dim)
        self.logvar = nn.Linear(in_features=hidden_dim, out_features=latent_dim
            )

    def forward(self, x):
        output = self.layers(x)
        mu = self.mu(output)
        logvar = self.logvar(output)
        return mu, logvar


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'item_dim': 4}]
