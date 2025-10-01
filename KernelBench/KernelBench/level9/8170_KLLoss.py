import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt as sqrt
from itertools import product as product


class KLLoss(nn.Module):
    """
    Kl-loss function for bounding box regression from CVPR 2019 paper:
    Bounding Box Regression with Uncertainty for Accurate Object Detection
    by Yihui He, Chenchen Zhu, Jianren Wang. Marios Savvides, Xiangyu Zhang

    It is a replacement for the Smooth L1 loss often used in bounding box regression.

    The regression loss for a coordinate depends on |xg − xe| > 1 or not:

    Loss |xg − xe| ≤ 1:

        Lreg1 ∝ e^{−α} * 1/2(xg − xe)^2 + 1/2α

    and if |xg − xe| > 1, Loss:

        Lreg2 = e^{−α} (|xg − xe| − 1/2) + 1/2α

    PyTorch implementation by Jasper Bakker (JappaB @github)
    """

    def __init__(self, loc_loss_weight=1.0):
        super(KLLoss, self).__init__()
        self.loc_loss_weight = loc_loss_weight

    def forward(self, xg, xe, alpha):
        """
        :param xg: The ground truth of the bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :param xe: The estimated bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :param alpha: The log(sigma^2) of the bounding box coordinates in x1y1x2y2 format
            shape: [number_of_boxes, 4]
        :return: total_kl_loss
        """
        assert xg.shape == xe.shape and xg.shape == alpha.shape, 'The shapes of the input tensors must be the same'
        smooth_l1 = F.smooth_l1_loss(xe, xg, reduction='none')
        exp_min_alpha = torch.exp(-alpha)
        half_alpha = 0.5 * alpha
        total_kl_loss = (exp_min_alpha * smooth_l1 + half_alpha).sum()
        return total_kl_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
