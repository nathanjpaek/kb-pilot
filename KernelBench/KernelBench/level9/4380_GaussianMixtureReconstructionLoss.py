import torch
import numpy as np
import torch as th


def gaussian_pdfs(dx, dy, params):
    """Returns the pdf at (dx, dy) for each Gaussian in the mixture.
    """
    dx = dx.unsqueeze(-1)
    dy = dy.unsqueeze(-1)
    mu_x = params[..., 0]
    mu_y = params[..., 1]
    sigma_x = params[..., 2].exp()
    sigma_y = params[..., 3].exp()
    rho_xy = th.tanh(params[..., 4])
    x = ((dx - mu_x) / sigma_x).pow(2)
    y = ((dy - mu_y) / sigma_y).pow(2)
    xy = (dx - mu_x) * (dy - mu_y) / (sigma_x * sigma_y)
    arg = x + y - 2.0 * rho_xy * xy
    pdf = th.exp(-arg / (2 * (1.0 - rho_xy.pow(2))))
    norm = 2.0 * np.pi * sigma_x * sigma_y * (1.0 - rho_xy.pow(2)).sqrt()
    return pdf / norm


class GaussianMixtureReconstructionLoss(th.nn.Module):
    """
    Args:
    """

    def __init__(self, eps=1e-05):
        super(GaussianMixtureReconstructionLoss, self).__init__()
        self.eps = eps

    def forward(self, pen_logits, mixture_logits, gaussian_params, targets):
        dx = targets[..., 0]
        dy = targets[..., 1]
        pen_state = targets[..., 2:].argmax(-1)
        valid_stroke = (targets[..., -1] != 1.0).float()
        mixture_weights = th.nn.functional.softmax(mixture_logits, -1)
        pdfs = gaussian_pdfs(dx, dy, gaussian_params)
        position_loss = -th.log(self.eps + (pdfs * mixture_weights).sum(-1))
        position_loss = (position_loss * valid_stroke).sum(
            ) / valid_stroke.sum()
        pen_loss = th.nn.functional.cross_entropy(pen_logits.view(-1, 3),
            pen_state.view(-1))
        return position_loss + pen_loss


def get_inputs():
    return [torch.rand([4, 3]), torch.rand([4, 4]), torch.rand([4, 5]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
