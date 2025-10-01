import torch
import torch.nn as nn
import torch.nn.functional as F


class Debayer2x2(nn.Module):
    """Demosaicing of Bayer images using 2x2 convolutions.

    Requires BG-Bayer color filter array layout. That is,
    the image[1,1]='B', image[1,2]='G'.
    """

    def __init__(self):
        super(Debayer2x2, self).__init__()
        self.kernels = nn.Parameter(torch.tensor([[1, 0], [0, 0], [0, 0.5],
            [0.5, 0], [0, 0], [0, 1]]).view(3, 1, 2, 2), requires_grad=False)

    def forward(self, x):
        """Debayer image.

        Parameters
        ----------
        x : Bx1xHxW tensor
            Images to debayer

        Returns
        -------
        rgb : Bx3xHxW tensor
            Color images in RGB channel order.
        """
        x = F.conv2d(x, self.kernels, stride=2)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners
            =False)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
