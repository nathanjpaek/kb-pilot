import torch
import torch.nn as nn


class run_latent(nn.Module):

    def __init__(self, in_dim, hidden_dim):
        super(run_latent, self).__init__()
        self.fc_z_mean = nn.Linear(in_dim, hidden_dim)
        self.fc_z_log_sigma = nn.Linear(in_dim, hidden_dim)
        self.fc_gen = nn.Linear(hidden_dim, in_dim)
        self.weights_init()

    def forward(self, x):
        z_mean = self.fc_z_mean(x)
        z_log_sigma_sq = self.fc_z_log_sigma(x)
        z = self.reparameterize(z_mean, z_log_sigma_sq)
        x_recon = self.fc_gen(z)
        x_recon = nn.functional.softmax(x_recon, dim=0)
        return x_recon, z_mean, z_log_sigma_sq

    def reparameterize(self, mu, log_var):
        std = torch.sqrt(torch.clamp(torch.exp(log_var), min=1e-10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def weights_init(self):
        nn.init.xavier_uniform_(self.fc_z_mean.weight)
        nn.init.constant_(self.fc_z_mean.bias, 0)
        nn.init.xavier_uniform_(self.fc_z_log_sigma.weight)
        nn.init.constant_(self.fc_z_log_sigma.bias, 0)
        nn.init.xavier_uniform_(self.fc_gen.weight)
        nn.init.constant_(self.fc_gen.bias, 0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'hidden_dim': 4}]
