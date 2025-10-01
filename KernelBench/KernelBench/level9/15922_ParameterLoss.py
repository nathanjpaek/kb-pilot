import torch
import torch.nn as nn


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        SMPL parameter loss module.
        """
        super(ParameterLoss, self).__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: 'torch.Tensor', gt_param: 'torch.Tensor',
        has_param: 'torch.Tensor'):
        """
        Compute SMPL parameter loss.
        Args:
            pred_param (torch.Tensor): Tensor of shape [B, S, ...] containing the predicted parameters (body pose / global orientation / betas)
            gt_param (torch.Tensor): Tensor of shape [B, S, ...] containing the ground truth SMPL parameters.
        Returns:
            torch.Tensor: L2 parameter loss loss.
        """
        batch_size = pred_param.shape[0]
        num_samples = pred_param.shape[1]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size, num_samples] + [1] * (num_dims - 2)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = has_param * self.loss_fn(pred_param, gt_param)
        return loss_param


def get_inputs():
    return [torch.rand([4, 4, 1, 1]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 1, 1])]


def get_init_inputs():
    return [[], {}]
