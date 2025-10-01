import torch
from torch import nn
import torch.nn.functional as F


class KDE(nn.Module):
    """KD on embeddings - KDE"""

    def __init__(self):
        super(KDE, self).__init__()

    def forward(self, embedding_s, embedding_t):
        inputs_embed = F.normalize(embedding_s, p=2.0, dim=1)
        targets_embed = F.normalize(embedding_t, p=2.0, dim=1)
        loss_kde = nn.MSELoss(reduction='sum')(inputs_embed, targets_embed)
        return loss_kde


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
