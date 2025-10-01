import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn as nn


class JointsCELoss(nn.Module):

    def __init__(self):
        super(JointsCELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        loss = 0
        for idx in range(num_joints):
            class_gt = target[:, idx].view(batch_size)
            class_pred = output[:, idx]
            loss += self.criterion(class_pred, class_gt)
        return loss / num_joints


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
