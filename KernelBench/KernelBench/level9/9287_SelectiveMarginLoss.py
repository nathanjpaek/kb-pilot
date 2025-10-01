import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data


class SelectiveMarginLoss(nn.Module):

    def __init__(self, loss_weight=5e-05, margin=0.2):
        super(SelectiveMarginLoss, self).__init__()
        self.margin = margin
        self.loss_weight = loss_weight

    def forward(self, pos_samples, neg_samples, has_sample):
        margin_diff = torch.clamp(pos_samples - neg_samples + self.margin,
            min=0, max=1000000.0)
        num_sample = max(torch.sum(has_sample), 1)
        return self.loss_weight * (torch.sum(margin_diff * has_sample) /
            num_sample)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
