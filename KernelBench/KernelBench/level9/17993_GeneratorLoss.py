import torch
from torch import nn


class GeneratorLoss(nn.Module):
    """
    Generator (BCE) loss function

    Args:
        alpha (default: int=1): Coefficient by which map loss will be multiplied
        beta (default: int=1): Coefficient by which point loss will be multiplied
    """

    def __init__(self, alpha=1, beta=1):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, fake_mpred, fake_ppred):
        fake_mtarget = torch.ones_like(fake_mpred)
        torch.ones_like(fake_ppred)
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        loss = self.alpha * map_loss + self.beta * point_loss
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
