import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Loss is proportional to square distance when inputs are of the same type, and proportional to
    the square of margin - distance when the classes are different. Margin is a user-specifiable
    hyperparameter.
    """

    def __init__(self, margin=2.0, pos_weight=0.5):
        super(ContrastiveLoss, self).__init__()
        self.pos_weight = pos_weight
        self.margin = margin

    def forward(self, distance, label):
        contrastive_loss = torch.mean((1 - self.pos_weight) * label * torch
            .pow(distance, 2) + self.pos_weight * (1 - label) * torch.pow(
            torch.clamp(self.margin - distance, min=0.0), 2))
        return contrastive_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
