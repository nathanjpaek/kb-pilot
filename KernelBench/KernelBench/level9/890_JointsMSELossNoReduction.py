import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class JointsMSELossNoReduction(nn.Module):

    def __init__(self, use_target_weight, logger):
        super(JointsMSELossNoReduction, self).__init__()
        self.criterion = lambda x, y: ((x - y) ** 2).sum(1).unsqueeze(1)
        self.use_target_weight = use_target_weight
        self.logger = logger

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1
            )
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                heatmap_pred = heatmap_pred.mul(target_weight[:, idx])
                loss_val = self.criterion(heatmap_pred, heatmap_gt.mul(
                    target_weight[:, idx]))
                loss.append(0.5 * loss_val)
            else:
                loss.append(0.5 * self.criterion(heatmap_pred, heatmap_gt))
        loss = torch.cat(loss, dim=1)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'use_target_weight': 4, 'logger': 4}]
