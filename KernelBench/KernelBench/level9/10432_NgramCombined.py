import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed


class NgramCombined(nn.Module):

    def __init__(self, n_gram):
        super(NgramCombined, self).__init__()
        self.n_gram = n_gram

    def forward(self, x):
        out = x
        if self.n_gram > 1:
            for i_gram in range(1, self.n_gram):
                out = F.pad(x.transpose(-1, -2), [i_gram, 0], mode=
                    'constant', value=0).transpose(-1, -2)[:, :-i_gram, :] + x
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_gram': 4}]
