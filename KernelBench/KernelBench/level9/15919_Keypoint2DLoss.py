import torch
import torch.nn as nn


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: 'str'='l1'):
        """
        2D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint2DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_2d: 'torch.Tensor', gt_keypoints_2d:
        'torch.Tensor') ->torch.Tensor:
        """
        Compute 2D reprojection loss on the keypoints.
        Args:
            pred_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 2] containing projected 2D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_2d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the ground truth 2D keypoints and confidence.
        Returns:
            torch.Tensor: 2D keypoint loss.
        """
        conf = gt_keypoints_2d[:, :, :, -1].unsqueeze(-1).clone()
        conf.shape[0]
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :,
            :, :-1])).sum(dim=(2, 3))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 3]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
