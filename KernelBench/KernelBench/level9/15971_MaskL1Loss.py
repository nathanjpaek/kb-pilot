import torch
import torch.nn as nn


class MaskL1Loss(nn.Module):
    """
    Loss from paper <Pose Guided Person Image Generation> Sec3.1 pose mask loss
    """

    def __init__(self, ratio=1):
        super(MaskL1Loss, self).__init__()
        self.criterion = nn.L1Loss()
        self.ratio = ratio

    def forward(self, generated_img, target_img, mask):
        pose_mask_l1 = self.criterion(generated_img * mask, target_img * mask)
        return self.criterion(generated_img, target_img
            ) + pose_mask_l1 * self.ratio


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
