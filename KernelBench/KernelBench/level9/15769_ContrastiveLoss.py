import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):

    def __init__(self, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()
        cost_s = torch.clamp((self.margin - diag).expand_as(scores) +
            scores, min=0)
        cost_im = torch.clamp((self.margin - diag.view(-1, 1)).expand_as(
            scores) + scores, min=0)
        diag_s = torch.diag(cost_s.diag())
        diag_im = torch.diag(cost_im.diag())
        cost_s = cost_s - diag_s
        cost_im = cost_im - diag_im
        return cost_s.sum() + cost_im.sum()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
