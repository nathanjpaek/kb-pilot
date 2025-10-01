import torch
from torch import nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class SimSiamLoss(nn.Module):

    def __init__(self, version='simplified'):
        super().__init__()
        self.ver = version

    def asymmetric_loss(self, p, z):
        if self.ver == 'original':
            z = z.detach()
            p = nn.functional.normalize(p, dim=1)
            z = nn.functional.normalize(z, dim=1)
            return -(p * z).sum(dim=1).mean()
        elif self.ver == 'simplified':
            z = z.detach()
            return -nn.functional.cosine_similarity(p, z, dim=-1).mean()

    def forward(self, p1, z2):
        loss1 = self.asymmetric_loss(p1, z2)
        return 0.5 * loss1


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
