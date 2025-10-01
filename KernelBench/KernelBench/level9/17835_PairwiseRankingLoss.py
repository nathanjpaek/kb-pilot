import torch
import torch.nn as nn


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise ranking loss
    """

    def __init__(self, margin):
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor1, anchor2, img_sentc, sent_imgc):
        cost_sent = torch.clamp(self.margin - anchor1 + img_sentc, min=0.0
            ).sum()
        cost_img = torch.clamp(self.margin - anchor2 + sent_imgc, min=0.0).sum(
            )
        loss = cost_sent + cost_img
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
