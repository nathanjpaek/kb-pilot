import torch
import torch.nn as nn
import torch.fft


class DWT(nn.Module):
    """
    2D Discrete Wavelet Transform as implemented in [1]_.

    References
    ----------

    .. [1] Liu, Pengju, et al. “Multi-Level Wavelet-CNN for Image Restoration.” ArXiv:1805.07071 [Cs], May 2018.
    arXiv.org, http://arxiv.org/abs/1805.07071.

    """

    def __init__(self):
        super().__init__()
        self.requires_grad = False

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
