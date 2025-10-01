import torch
import torch.nn as nn


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: 'str'='l1'):
        """
        3D keypoint loss module.
        Args:
            loss_type (str): Choose between l1 and l2 losses.
        """
        super(Keypoint3DLoss, self).__init__()
        if loss_type == 'l1':
            self.loss_fn = nn.L1Loss(reduction='none')
        elif loss_type == 'l2':
            self.loss_fn = nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError('Unsupported loss function')

    def forward(self, pred_keypoints_3d: 'torch.Tensor', gt_keypoints_3d:
        'torch.Tensor', pelvis_id: 'int'=39):
        """
        Compute 3D keypoint loss.
        Args:
            pred_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 3] containing the predicted 3D keypoints (B: batch_size, S: num_samples, N: num_keypoints)
            gt_keypoints_3d (torch.Tensor): Tensor of shape [B, S, N, 4] containing the ground truth 3D keypoints and confidence.
        Returns:
            torch.Tensor: 3D keypoint loss.
        """
        pred_keypoints_3d.shape[0]
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, :,
            pelvis_id, :].unsqueeze(dim=2)
        gt_keypoints_3d[:, :, :, :-1] = gt_keypoints_3d[:, :, :, :-1
            ] - gt_keypoints_3d[:, :, pelvis_id, :-1].unsqueeze(dim=2)
        conf = gt_keypoints_3d[:, :, :, -1].unsqueeze(-1).clone()
        gt_keypoints_3d = gt_keypoints_3d[:, :, :, :-1]
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d)).sum(
            dim=(2, 3))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 40, 3]), torch.rand([4, 4, 40, 4])]


def get_init_inputs():
    return [[], {}]
