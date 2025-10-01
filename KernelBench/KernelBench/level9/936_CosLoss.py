import torch
from torch import nn
import torch.utils.data


class CosLoss(nn.Module):

    def __init__(self, factor=6e-07, havesum=True, havemax=True):
        super(CosLoss, self).__init__()
        self.factor = factor
        self.havesum = havesum
        self.havemax = havemax

    def forward(self, w):
        mask = torch.ones_like(w)
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                mask[i, j, j] = -1
        nw = mask * w
        tmp, _ = torch.max(nw, dim=1)
        tmp, _ = torch.max(tmp, dim=1)
        if self.havesum and self.havemax:
            tmp_all = tmp + self.factor * torch.sum(torch.sum(nw, dim=1), dim=1
                )
        elif self.havesum:
            tmp_all = self.factor * torch.sum(torch.sum(nw, dim=1), dim=1)
        else:
            tmp_all = tmp
        loss = torch.mean(tmp_all)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
