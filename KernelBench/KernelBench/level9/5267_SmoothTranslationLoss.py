import torch
import torch.nn as nn
import torch.cuda.comm


class SmoothTranslationLoss(nn.Module):
    """Smooth loss for translations.

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

    def forward(self, translation, loss_weight_override=None,
        reduction_override=None):
        """Forward function of loss.

        Args:
            translation (torch.Tensor): The body translation parameters
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
        translation_diff = translation[1:] - translation[:-1]
        smooth_translation_loss = translation_diff.abs().sum(dim=-1,
            keepdim=True)
        smooth_translation_loss = torch.cat([torch.zeros_like(
            smooth_translation_loss)[:1], smooth_translation_loss]).sum(dim=-1)
        smooth_translation_loss *= 1000.0
        smooth_translation_loss = loss_weight * smooth_translation_loss
        if reduction == 'mean':
            smooth_translation_loss = smooth_translation_loss.mean()
        elif reduction == 'sum':
            smooth_translation_loss = smooth_translation_loss.sum()
        return smooth_translation_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
