import torch
import torch.nn as nn


def get_outnorm(x: 'torch.Tensor', out_norm: 'str'='') ->torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    img_shape = x.shape
    if not out_norm:
        return 1
    norm = 1
    if 'b' in out_norm:
        norm /= img_shape[0]
    if 'c' in out_norm:
        norm /= img_shape[-3]
    if 'i' in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-06, out_norm: 'str'='bci'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
        self.out_norm = out_norm

    def forward(self, x, y):
        norm = get_outnorm(x, self.out_norm)
        loss = torch.sum(torch.sqrt((x - y).pow(2) + self.eps ** 2))
        return loss * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
