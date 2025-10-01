import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from typing import *


class PFLDLoss(nn.Module):
    """Weighted loss of L2 distance with the pose angle for PFLD."""

    def __init__(self):
        super(PFLDLoss, self).__init__()

    def forward(self, landmark_gt, euler_angle_gt, angle, landmarks):
        """
        Calculate weighted L2 loss for PFLD.

        Parameters
        ----------
        landmark_gt : tensor
            the ground truth of landmarks
        euler_angle_gt : tensor
            the ground truth of pose angle
        angle : tensor
            the predicted pose angle
        landmarks : float32
            the predicted landmarks

        Returns
        -------
        output: tensor
            the weighted L2 loss
        output: tensor
            the normal L2 loss
        """
        weight_angle = torch.sum(1 - torch.cos(angle - euler_angle_gt), axis=1)
        l2_distant = torch.sum((landmark_gt - landmarks) ** 2, axis=1)
        return torch.mean(weight_angle * l2_distant), torch.mean(l2_distant)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
