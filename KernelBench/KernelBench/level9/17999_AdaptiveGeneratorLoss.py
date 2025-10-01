import torch
from torch import nn


class AdaptiveGeneratorLoss(nn.Module):
    """
    Adaptive Generator (BCE) loss function (depends on losses of Discriminators)

    Args:
        alpha (default: int=3): Coefficient for map and point losses
    """

    def __init__(self, alpha=3):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, fake_mpred, fake_ppred, d_mloss, d_ploss):
        fake_mtarget = torch.ones_like(fake_mpred)
        torch.ones_like(fake_ppred)
        map_loss = self.adv_criterion(fake_mpred, fake_mtarget)
        point_loss = self.adv_criterion(fake_ppred, fake_mpred)
        map_coef = self.alpha * d_mloss / (d_ploss + self.alpha * d_mloss)
        point_coef = d_ploss / (d_ploss + self.alpha * d_mloss)
        loss = map_coef * map_loss + point_coef * point_loss
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
