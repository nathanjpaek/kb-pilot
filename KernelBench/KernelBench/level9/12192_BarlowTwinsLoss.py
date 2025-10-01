import torch
from torch import nn
from torch.nn.modules.loss import *
from torch.nn.modules import *
from torch.optim import *
from torch.optim.lr_scheduler import *
import torch.backends


class BarlowTwinsLoss(nn.Module):
    """The Contrastive embedding loss.

    It has been proposed in `Barlow Twins:
    Self-Supervised Learning via Redundancy Reduction`_.

    Example:

    .. code-block:: python

        import torch
        from torch.nn import functional as F
        from catalyst.contrib.nn import BarlowTwinsLoss

        embeddings_left = F.normalize(torch.rand(256, 64, requires_grad=True))
        embeddings_right = F.normalize(torch.rand(256, 64, requires_grad=True))
        criterion = BarlowTwinsLoss(offdiag_lambda = 1)
        criterion(embeddings_left, embeddings_right)

    .. _`Barlow Twins: Self-Supervised Learning via Redundancy Reduction`:
        https://arxiv.org/abs/2103.03230
    """

    def __init__(self, offdiag_lambda=1.0, eps=1e-12):
        """
        Args:
            offdiag_lambda: trade-off parameter
            eps: shift for the varience (var + eps)
        """
        super().__init__()
        self.offdiag_lambda = offdiag_lambda
        self.eps = eps

    def forward(self, embeddings_left: 'torch.Tensor', embeddings_right:
        'torch.Tensor') ->torch.Tensor:
        """Forward propagation method for the contrastive loss.

        Args:
            embeddings_left: left objects embeddings [batch_size, features_dim]
            embeddings_right: right objects embeddings [batch_size, features_dim]

        Raises:
            ValueError: if the batch size is 1
            ValueError: if embeddings_left and embeddings_right shapes are different
            ValueError: if embeddings shapes are not in a form (batch_size, features_dim)

        Returns:
            torch.Tensor: loss
        """
        shape_left, shape_right = embeddings_left.shape, embeddings_right.shape
        if len(shape_left) != 2:
            raise ValueError(
                f'Left shape should be (batch_size, feature_dim), but got - {shape_left}!'
                )
        elif len(shape_right) != 2:
            raise ValueError(
                f'Right shape should be (batch_size, feature_dim), but got - {shape_right}!'
                )
        if shape_left[0] == 1:
            raise ValueError(
                f'Batch size should be >= 2, but got - {shape_left[0]}!')
        if shape_left != shape_right:
            raise ValueError(
                f'Shapes should be equall, but got - {shape_left} and {shape_right}!'
                )
        z_left = (embeddings_left - embeddings_left.mean(dim=0)) / (
            embeddings_left.var(dim=0) + self.eps).pow(1 / 2)
        z_right = (embeddings_right - embeddings_right.mean(dim=0)) / (
            embeddings_right.var(dim=0) + self.eps).pow(1 / 2)
        batch_size = z_left.shape[0]
        cross_correlation = torch.matmul(z_left.T, z_right) / batch_size
        on_diag = torch.diagonal(cross_correlation)
        off_diag = cross_correlation.clone().fill_diagonal_(0)
        on_diag_loss = on_diag.add_(-1).pow_(2).sum()
        off_diag_loss = off_diag.pow_(2).sum()
        loss = on_diag_loss + self.offdiag_lambda * off_diag_loss
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
