import torch
from torch import nn
import torch.nn.functional as F


def angle(pred):
    """Calculate the angle-wise relational potential which measures the angle
    formed by the three examples in the output representation space.

    Args:
        pred (torch.Tensor): The prediction of the teacher or student with
            shape (N, C).
    """
    pred_vec = pred.unsqueeze(0) - pred.unsqueeze(1)
    norm_pred_vec = F.normalize(pred_vec, p=2, dim=2)
    angle = torch.bmm(norm_pred_vec, norm_pred_vec.transpose(1, 2)).view(-1)
    return angle


class AngleWiseRKD(nn.Module):
    """PyTorch version of angle-wise loss of `Relational Knowledge
    Distillation.

    <https://arxiv.org/abs/1904.05068>`_.

    Args:
        loss_weight (float): Weight of angle-wise distillation loss.
            Defaults to 50.0.
        with_l2_norm (bool): Whether to normalize the model predictions before
            calculating the loss. Defaults to True.
    """

    def __init__(self, loss_weight=50.0, with_l2_norm=True):
        super(AngleWiseRKD, self).__init__()
        self.loss_weight = loss_weight
        self.with_l2_norm = with_l2_norm

    def angle_loss(self, preds_S, preds_T):
        """Calculate the angle-wise distillation loss."""
        angle_T = angle(preds_T)
        angle_S = angle(preds_S)
        return F.smooth_l1_loss(angle_S, angle_T)

    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W) or shape (N, C).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W) or shape (N, C).
        Return:
            torch.Tensor: The calculated loss value.
        """
        preds_S = preds_S.view(preds_S.shape[0], -1)
        preds_T = preds_T.view(preds_T.shape[0], -1)
        if self.with_l2_norm:
            preds_S = F.normalize(preds_S, p=2, dim=-1)
            preds_T = F.normalize(preds_T, p=2, dim=-1)
        loss = self.angle_loss(preds_S, preds_T) * self.loss_weight
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
