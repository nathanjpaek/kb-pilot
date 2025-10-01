import torch
from torch import nn


class DiscriminatorLoss(nn.Module):
    """
    Discriminator (BCE) loss function

    Args:
        - None -
    """

    def __init__(self):
        super().__init__()
        self.adv_criterion = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.adv_criterion(fake_pred, fake_target)
        real_loss = self.adv_criterion(real_pred, real_target)
        loss = (fake_loss + real_loss) / 2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
