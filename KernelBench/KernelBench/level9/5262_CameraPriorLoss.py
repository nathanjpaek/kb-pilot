import torch
import torch.nn as nn
import torch.cuda.comm


class CameraPriorLoss(nn.Module):
    """Prior loss for predicted camera.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        scale (float, optional): The scale coefficient for regularizing camera
            parameters. Defaults to 10
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, scale=10, reduction='mean', loss_weight=1.0):
        super().__init__()
        self.scale = scale
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, cameras, loss_weight_override=None,
        reduction_override=None):
        """Forward function of loss.

        Args:
            cameras (torch.Tensor): The predicted camera parameters
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
        camera_prior_loss = torch.exp(-cameras[:, 0] * self.scale)
        camera_prior_loss = torch.pow(camera_prior_loss, 2) * loss_weight
        if reduction == 'mean':
            camera_prior_loss = camera_prior_loss.mean()
        elif reduction == 'sum':
            camera_prior_loss = camera_prior_loss.sum()
        return camera_prior_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
