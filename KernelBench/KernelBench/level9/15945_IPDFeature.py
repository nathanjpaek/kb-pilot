import math
import torch
import torch as th
import torch.nn as nn


class IPDFeature(nn.Module):
    """
    Compute inter-channel phase difference
    """

    def __init__(self, ipd_index='1,0;2,0;3,0;4,0;5,0;6,0', cos=True, sin=False
        ):
        super(IPDFeature, self).__init__()

        def split_index(sstr):
            return [tuple(map(int, p.split(','))) for p in sstr.split(';')]
        pair = split_index(ipd_index)
        self.index_l = [t[0] for t in pair]
        self.index_r = [t[1] for t in pair]
        self.ipd_index = ipd_index
        self.cos = cos
        self.sin = sin
        self.num_pairs = len(pair) * 2 if cos and sin else len(pair)

    def extra_repr(self):
        return f'ipd_index={self.ipd_index}, cos={self.cos}, sin={self.sin}'

    def forward(self, p):
        """
        Accept multi-channel phase and output inter-channel phase difference
        args
            p: phase matrix, N x C x F x T
        return
            ipd: N x MF x T
        """
        if p.dim() not in [3, 4]:
            raise RuntimeError('{} expect 3/4D tensor, but got {:d} instead'
                .format(self.__name__, p.dim()))
        if p.dim() == 3:
            p = p.unsqueeze(0)
        N, _, _, T = p.shape
        pha_dif = p[:, self.index_l] - p[:, self.index_r]
        if self.cos:
            ipd = th.cos(pha_dif)
            if self.sin:
                ipd = th.cat([ipd, th.sin(pha_dif)], 2)
        else:
            ipd = th.fmod(pha_dif, 2 * math.pi) - math.pi
        ipd = ipd.view(N, -1, T)
        return ipd


def get_inputs():
    return [torch.rand([7, 4, 4])]


def get_init_inputs():
    return [[], {}]
