import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden2mean = nn.Linear(input_dim, latent_dim)
        self.hidden2logv = nn.Linear(input_dim, latent_dim)
        self.latent2hidden = nn.Linear(latent_dim, input_dim)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.hidden2mean.weight)
        nn.init.xavier_uniform_(self.hidden2logv.weight)
        nn.init.xavier_uniform_(self.latent2hidden.weight)

    def forward(self, h):
        mean = self.hidden2mean(h)
        logv = self.hidden2logv(h)
        std = torch.exp(0.5 * logv)
        z = torch.randn((h.size(0), self.latent_dim), device=h.device)
        z = z * std + mean
        restruct_hidden = self.latent2hidden(z)
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()
            ) / logv.size(0)
        dist_loss = F.mse_loss(restruct_hidden, h)
        return dist_loss, kl_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'latent_dim': 4}]
