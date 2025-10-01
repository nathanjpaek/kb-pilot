import torch
import torch.nn as nn
import torch.cuda.comm


class ShapePriorLoss(nn.Module):
    """Prior loss for body shape parameters.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, betas, loss_weight_override=None, reduction_override=None
        ):
        """Forward function of loss.

        Args:
            betas (torch.Tensor): The body shape parameters
            loss_weight_override (float, optional): The weight of loss used to
                override the original weight of loss
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (reduction_override if reduction_override else self.
            reduction)
        loss_weight = (loss_weight_override if loss_weight_override is not
            None else self.loss_weight)
        shape_prior_loss = loss_weight * betas ** 2
        if reduction == 'mean':
            shape_prior_loss = shape_prior_loss.mean()
        elif reduction == 'sum':
            shape_prior_loss = shape_prior_loss.sum()
        return shape_prior_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
