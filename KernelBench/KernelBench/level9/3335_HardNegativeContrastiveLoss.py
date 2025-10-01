import torch
import torch.nn as nn


class HardNegativeContrastiveLoss(nn.Module):

    def __init__(self, nmax=1, margin=0.2):
        super(HardNegativeContrastiveLoss, self).__init__()
        self.margin = margin
        self.nmax = nmax

    def forward(self, imgs, caps):
        scores = torch.mm(imgs, caps.t())
        diag = scores.diag()
        scores = scores - 2 * torch.diag(scores.diag())
        sorted_cap, _ = torch.sort(scores, 0, descending=True)
        sorted_img, _ = torch.sort(scores, 1, descending=True)
        max_c = sorted_cap[:self.nmax, :]
        max_i = sorted_img[:, :self.nmax]
        neg_cap = torch.sum(torch.clamp(max_c + (self.margin - diag).view(1,
            -1).expand_as(max_c), min=0))
        neg_img = torch.sum(torch.clamp(max_i + (self.margin - diag).view(-
            1, 1).expand_as(max_i), min=0))
        loss = neg_cap + neg_img
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
