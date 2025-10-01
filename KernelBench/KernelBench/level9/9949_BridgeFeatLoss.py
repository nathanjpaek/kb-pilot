import torch
import torch.nn as nn
import torch.utils.data


class BridgeFeatLoss(nn.Module):
    """Bridge loss on feature space.
    """

    def __init__(self):
        super(BridgeFeatLoss, self).__init__()

    def forward(self, f_s, f_t, f_mixed, lam):
        dist_mixed2s = ((f_mixed - f_s) ** 2).sum(1, keepdim=True)
        dist_mixed2t = ((f_mixed - f_t) ** 2).sum(1, keepdim=True)
        dist_mixed2s = dist_mixed2s.clamp(min=1e-12).sqrt()
        dist_mixed2t = dist_mixed2t.clamp(min=1e-12).sqrt()
        dist_mixed = torch.cat((dist_mixed2s, dist_mixed2t), 1)
        lam_dist_mixed = (lam * dist_mixed).sum(1, keepdim=True)
        loss = lam_dist_mixed.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
