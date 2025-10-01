import torch
import torch.nn as nn
import torch.nn.functional as F


def layer_init(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class EnvModel(nn.Module):

    def __init__(self, phi_dim, action_dim):
        super(EnvModel, self).__init__()
        self.hidden_dim = 300
        self.fc_r1 = layer_init(nn.Linear(phi_dim + action_dim, self.
            hidden_dim))
        self.fc_r2 = layer_init(nn.Linear(self.hidden_dim, 1))
        self.fc_t1 = layer_init(nn.Linear(phi_dim, phi_dim))
        self.fc_t2 = layer_init(nn.Linear(phi_dim + action_dim, phi_dim))

    def forward(self, phi_s, action):
        phi = torch.cat([phi_s, action], dim=-1)
        r = self.fc_r2(F.tanh(self.fc_r1(phi)))
        phi_s_prime = phi_s + F.tanh(self.fc_t1(phi_s))
        phi_sa_prime = torch.cat([phi_s_prime, action], dim=-1)
        phi_s_prime = phi_s_prime + F.tanh(self.fc_t2(phi_sa_prime))
        return phi_s_prime, r


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'phi_dim': 4, 'action_dim': 4}]
