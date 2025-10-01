import torch
import torch.nn as nn


def ortho(w: 'torch.Tensor') ->torch.Tensor:
    """
    Returns the orthogonal loss for weight matrix `m`, from Big GAN.

    https://arxiv.org/abs/1809.11096

    :math:`R_{\\beta}(W)= ||W^T W  \\odot (1 - I)||_F^2`
    """
    cosine = torch.einsum('ij,ji->ij', w, w)
    no_diag = 1 - torch.eye(w.shape[0], device=w.device)
    return (cosine * no_diag).pow(2).sum(dim=1).mean()


class OrthoLoss(nn.Module):
    """
    Orthogonal loss

    See :func:`torchelie.loss.ortho` for details.
    """

    def forward(self, w):
        return ortho(w)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
