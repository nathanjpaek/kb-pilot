import torch
import torch.nn.functional as F
import torch.nn as nn


class SoftDetectionModule(nn.Module):

    def __init__(self, soft_local_max_size=3):
        super(SoftDetectionModule, self).__init__()
        self.soft_local_max_size = soft_local_max_size
        self.pad = self.soft_local_max_size // 2

    def forward(self, batch):
        b = batch.size(0)
        batch = F.relu(batch)
        max_per_sample = torch.max(batch.view(b, -1), dim=1)[0]
        exp = torch.exp(batch / max_per_sample.view(b, 1, 1, 1))
        sum_exp = self.soft_local_max_size ** 2 * F.avg_pool2d(F.pad(exp, [
            self.pad] * 4, mode='constant', value=1.0), self.
            soft_local_max_size, stride=1)
        local_max_score = exp / sum_exp
        depth_wise_max = torch.max(batch, dim=1)[0]
        depth_wise_max_score = batch / depth_wise_max.unsqueeze(1)
        all_scores = local_max_score * depth_wise_max_score
        score = torch.max(all_scores, dim=1)[0]
        score = score / (torch.sum(score.view(b, -1), dim=1).view(b, 1, 1) +
            1e-05)
        return score


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
