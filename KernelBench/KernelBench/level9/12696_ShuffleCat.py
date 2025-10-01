import torch
import torch.nn as nn


class ShuffleCat(nn.Module):

    def forward(self, a, b):
        assert a.size() == b.size()
        n, c, h, w = a.size()
        a = a.permute(0, 2, 3, 1).contiguous().view(-1, c)
        b = b.permute(0, 2, 3, 1).contiguous().view(-1, c)
        x = torch.cat((a, b), dim=0).transpose(1, 0).contiguous()
        x = x.view(c * 2, n, h, w).permute(1, 0, 2, 3)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
