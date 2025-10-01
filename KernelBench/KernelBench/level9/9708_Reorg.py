import torch
import torch.nn as nn


class Reorg(nn.Module):
    dump_patches = True

    def __init__(self):
        super(Reorg, self).__init__()

    def forward(self, x):
        ss = x.size()
        out = x.view(ss[0], ss[1], ss[2] // 2, 2, ss[3]).view(ss[0], ss[1],
            ss[2] // 2, 2, ss[3] // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous(
            ).view(ss[0], -1, ss[2] // 2, ss[3] // 2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
