import torch
import torch.utils.data
import torch.nn as nn


class JointMseLoss(nn.Module):

    def __init__(self):
        super(JointMseLoss, self).__init__()
        self.mseLoss = nn.MSELoss()

    def forward(self, pre1, pre2, gt, sobel_gt):
        loss1 = self.mseLoss(pre1, sobel_gt)
        loss2 = self.mseLoss(pre2, gt)
        loss = loss1 + loss2
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
