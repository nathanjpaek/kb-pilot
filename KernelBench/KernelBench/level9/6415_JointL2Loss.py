import torch
import torch.nn as nn
import torch.utils.data


class JointL2Loss(nn.Module):

    def __init__(self):
        super(JointL2Loss, self).__init__()

    def forward(self, joint_pred, joint_gt):
        batch_size, joint_num, _ = joint_gt.shape
        joint_pred = joint_pred.view(batch_size * joint_num, -1)
        joint_gt = joint_gt.view(batch_size * joint_num, -1)
        offset = torch.sum(torch.pow(joint_gt - joint_pred, 2), dim=1)
        return torch.sqrt(offset).mean()


def get_inputs():
    return [torch.rand([4, 16, 4, 4]), torch.rand([4, 64, 4])]


def get_init_inputs():
    return [[], {}]
