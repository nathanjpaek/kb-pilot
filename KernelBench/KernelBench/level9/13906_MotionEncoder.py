import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    """
    Encodes motion features from the correlation levels of the pyramid
    and the input flow estimate using convolution layers.


    Parameters
    ----------
    corr_radius : int
        Correlation radius of the correlation pyramid
    corr_levels : int
        Correlation levels of the correlation pyramid

    """

    def __init__(self, corr_radius, corr_levels):
        super(MotionEncoder, self).__init__()
        cor_planes = corr_levels * (2 * corr_radius + 1) ** 2
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        """
        Parameters
        ----------
        flow : torch.Tensor
            A tensor of shape N x 2 x H x W

        corr : torch.Tensor
            A tensor of shape N x (corr_levels * (2 * corr_radius + 1) ** 2) x H x W

        Returns
        -------
        torch.Tensor
            A tensor of shape N x 128 x H x W
        """
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


def get_inputs():
    return [torch.rand([4, 2, 64, 64]), torch.rand([4, 324, 64, 64])]


def get_init_inputs():
    return [[], {'corr_radius': 4, 'corr_levels': 4}]
