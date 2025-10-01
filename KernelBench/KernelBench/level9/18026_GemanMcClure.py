import torch
import torch.optim


class L2(torch.nn.Module):

    def __init__(self):
        super(L2, self).__init__()

    def forward(self, pred: 'torch.Tensor', gt: 'torch.Tensor'=None,
        weights: 'torch.Tensor'=None, mask: 'torch.Tensor'=None
        ) ->torch.Tensor:
        l2 = (gt - pred) ** 2 if gt is not None else pred ** 2
        if weights is not None:
            l2 = l2 * weights
        if mask is not None:
            l2 = l2[mask]
        return l2


class GemanMcClure(L2):
    """Implements the Geman-McClure error function.

    """

    def __init__(self, rho: 'float'=1.0):
        super(GemanMcClure, self).__init__()
        self.rho_sq = rho ** 2

    def forward(self, gt: 'torch.Tensor', pred: 'torch.Tensor', weights:
        'torch.Tensor'=None, mask: 'torch.Tensor'=None) ->torch.Tensor:
        L2 = super(GemanMcClure, self).forward(gt, pred)
        gm = L2 / (L2 + self.rho_sq) * self.rho_sq
        if weights is not None:
            gm = gm * weights
        if mask is not None:
            gm = gm[mask]
        return gm


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
