import torch
import torch.nn as nn
import torch.utils.cpp_extension


class Subspace(nn.Module):

    def __init__(self, latent_dim, channels, resolution):
        super().__init__()
        self.U = nn.Parameter(torch.empty(latent_dim, channels, resolution,
            resolution))
        nn.init.orthogonal_(self.U)
        l_init = [[(3.0 * i) for i in range(latent_dim, 0, -1)]]
        self.L = nn.Parameter(torch.tensor(l_init))
        self.mu = nn.Parameter(torch.zeros(1, channels, resolution, resolution)
            )

    def forward(self, z):
        x = (self.L * z)[:, :, None, None, None]
        x = self.U[None, ...] * x
        x = x.sum(1)
        x = x + self.mu
        return x

    def gram_schimdt(self, vector):
        """this doesn't work.
            It stops by OOM.
        """
        basis = vector[0:1] / vector[0:1].norm()
        for i in range(1, vector.size(0)):
            v = vector[i:i + 1]
            w = v - torch.mm(torch.mm(v, basis.T), basis)
            w = w / w.norm()
            basis = torch.cat([basis, w], dim=0)
        return basis


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'latent_dim': 4, 'channels': 4, 'resolution': 4}]
