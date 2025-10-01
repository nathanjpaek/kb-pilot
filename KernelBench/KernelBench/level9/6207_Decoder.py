import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """ Encoder
    """

    def __init__(self, n_levels, n_color, n_eccentricity, n_azimuth,
        n_theta, n_phase):
        super(Decoder, self).__init__()
        self.n_levels = n_levels
        self.n_color = n_color
        self.n_eccentricity = n_eccentricity
        self.n_azimuth = n_azimuth
        self.n_theta = n_theta
        self.n_phase = n_phase
        self.h_size = (n_levels * n_color * n_eccentricity * n_azimuth *
            n_theta * n_phase)

    def forward(self, x, theta=None):
        x = x.view(-1, self.n_color * self.n_theta * self.n_phase, self.
            n_levels * self.n_eccentricity, self.n_azimuth)
        lim = self.n_levels * self.n_eccentricity // 2
        x_int = x[:, :, :lim, ...]
        x_ext = x[:, :, lim:, ...]
        x_list = []
        for x in (x_int, x_ext):
            if theta is not None:
                theta_inv = theta
                theta_inv[:, :, 2] = -theta[:, :, 2].detach()
                grid = F.affine_grid(theta_inv, x.size())
                x = F.grid_sample(x, grid)
            x = x.view(-1, self.n_color, self.n_theta, self.n_phase, self.
                n_levels, self.n_eccentricity // 2, self.n_azimuth)
            x = x.permute(0, 4, 1, 5, 6, 2, 3).contiguous()
            x_list.append(x)
        x = torch.cat(x_list, 3)
        return x


def get_inputs():
    return [torch.rand([4, 64, 16, 4])]


def get_init_inputs():
    return [[], {'n_levels': 4, 'n_color': 4, 'n_eccentricity': 4,
        'n_azimuth': 4, 'n_theta': 4, 'n_phase': 4}]
