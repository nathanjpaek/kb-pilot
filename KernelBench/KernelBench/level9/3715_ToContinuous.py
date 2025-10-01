import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class ToContinuous(nn.Module):

    def __init__(self):
        super(ToContinuous, self).__init__()

    def forward(self, x):
        """
        :param x: tensor with dimension opt(batch x _ x bins x H x W
        :return:
        """
        assert len(x.shape) >= 3 and x.shape[-3] >= 2
        *other_dims, C, H, W = x.shape
        x = x.reshape(-1, C, H, W)
        x = torch.max(x, dim=1).indices
        sil = x > 0
        sil_float = sil.float()
        x = (x.float() - 1) / (C - 2)
        x = 2 * x - 1
        x[~sil] = -1
        x = torch.stack((x, sil_float), dim=0).permute(1, 0, 2, 3).contiguous()
        x = x.reshape(other_dims + [2, H, W])
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
