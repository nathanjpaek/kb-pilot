import torch


class DisentangledAELatent(torch.nn.Module):
    """Dense Dientangled Latent Layer between encoder and decoder"""

    def __init__(self, hidden_size: 'int', latent_size: 'int', dropout: 'float'
        ):
        super(DisentangledAELatent, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.latent = torch.nn.Linear(self.hidden_size, self.latent_size * 2)

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        return mu

    def forward(self, x, training=True):
        z_variables = self.latent(x)
        mu, logvar = torch.chunk(z_variables, 2, dim=1)
        z = self.reparameterize(mu, logvar, training=training)
        return z, mu, logvar


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'latent_size': 4, 'dropout': 0.5}]
