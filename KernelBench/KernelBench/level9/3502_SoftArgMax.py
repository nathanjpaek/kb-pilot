import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgMax(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, labels, kernel_size=0):
        """
        Args
            x: [B, C, Nd]
            labels: [Nd]
        Returns
            [B, C]
        """
        y = x * labels
        kernel_size = kernel_size if kernel_size > 0 else x.size(-1)
        x = F.avg_pool1d(x, kernel_size=kernel_size) * kernel_size
        y = F.avg_pool1d(y, kernel_size=kernel_size) * kernel_size
        y = y / (x + 1e-08)
        ind = x.argmax(dim=-1).unsqueeze(-1)
        res = torch.gather(y, dim=-1, index=ind)
        res = res.squeeze(-1)
        return res


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
