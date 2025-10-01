import torch
import torch.utils.data
import torch
import torch.nn as nn


class KeypointsMSESmoothLoss(nn.Module):

    def __init__(self, threshold=400):
        super().__init__()
        self.threshold = threshold

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        dimension = heatmaps_pred.shape[-1]
        diff = (heatmaps_gt - heatmaps_pred) ** 2 * target_weight[..., None]
        diff[diff > self.threshold] = torch.pow(diff[diff > self.threshold],
            0.1) * self.threshold ** 0.9
        loss = torch.sum(diff) / (dimension * max(1, torch.sum(target_weight)))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
