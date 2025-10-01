import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, class_num, feature_num, alpha=0.5):
        super(CenterLoss, self).__init__()
        self.class_num = class_num
        self.feature_num = feature_num
        self.class_centers = nn.Parameter(torch.randn(self.class_num, self.
            feature_num))

    def forward(self, embedding_batch, label_batch):
        label_center_batch = self.class_centers[label_batch]
        diff_batch = embedding_batch - label_center_batch
        loss = (diff_batch ** 2.0).sum(dim=1).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'class_num': 4, 'feature_num': 4}]
