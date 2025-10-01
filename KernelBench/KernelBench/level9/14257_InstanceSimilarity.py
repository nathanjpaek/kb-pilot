import torch
import torch.nn.functional as F
import torch.nn as nn


class InstanceSimilarity(nn.Module):
    """
    Instance Similarity based loss
    """

    def __init__(self, mse=True):
        super(InstanceSimilarity, self).__init__()
        self.mse = mse

    def _loss(self, fm_s, fm_t):
        fm_s = fm_s.view(fm_s.size(0), -1)
        G_s = torch.mm(fm_s, fm_s.t())
        norm_G_s = F.normalize(G_s, p=2, dim=1)
        fm_t = fm_t.view(fm_t.size(0), -1)
        G_t = torch.mm(fm_t, fm_t.t())
        norm_G_t = F.normalize(G_t, p=2, dim=1)
        loss = F.mse_loss(norm_G_s, norm_G_t) if self.mse else F.l1_loss(
            norm_G_s, norm_G_t)
        return loss

    def forward(self, g_s, g_t):
        return sum(self._loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
