import torch
import torch.utils.data
from torch import nn


class AELoss(nn.Module):

    def __init__(self, pull_factor, push_factor, distance, margin_push):
        super(AELoss, self).__init__()
        self.pull_factor = pull_factor
        self.push_factor = push_factor
        self.distance = distance
        self.margin_push = margin_push

    def forward(self, lof_tag_img, lof_tag_avg_img, lof_tag_avg_gather_img,
        mask, centerness_img=None):
        lof_tag_avg_gather_img = torch.round(lof_tag_avg_gather_img / self.
            distance) * self.distance
        tag = torch.pow(lof_tag_img - torch.round(lof_tag_avg_gather_img), 2)
        dist = lof_tag_avg_img.unsqueeze(0) - lof_tag_avg_img.unsqueeze(1)
        dist = self.distance + self.margin_push - torch.abs(dist)
        dist = nn.functional.relu(dist, inplace=True)
        dist = dist[mask]
        if centerness_img is not None:
            pull = (tag * centerness_img).sum() / centerness_img.sum()
            push = torch.zeros_like(pull)
            if mask.any():
                push = dist.sum() / mask.sum().float()
        else:
            pull = tag.mean()
            push = dist.mean()
        return self.pull_factor * pull, self.push_factor * push


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'pull_factor': 4, 'push_factor': 4, 'distance': 4,
        'margin_push': 4}]
