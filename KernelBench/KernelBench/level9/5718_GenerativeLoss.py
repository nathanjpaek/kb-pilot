import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class GenerativeLoss(nn.Module):

    def __init__(self):
        super(GenerativeLoss, self).__init__()
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, output, target):
        num_joints = output.shape[1]
        loss = 0
        for idx in range(num_joints):
            real_or_fake_pred = output[:, idx]
            real_or_fake_gt = target[:, idx]
            loss += self.criterion(real_or_fake_pred, real_or_fake_gt)
        return loss / num_joints


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
