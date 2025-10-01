import torch
import torch.utils.data
import torch.nn as nn
from torchvision.transforms import *


class OhemSphereLoss(nn.Module):

    def __init__(self, in_feats, n_classes, thresh=0.7, scale=14, *args, **
        kwargs):
        super(OhemSphereLoss, self).__init__(*args, **kwargs)
        self.thresh = thresh
        self.scale = scale
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.W = torch.nn.Parameter(torch.randn(in_feats, n_classes),
            requires_grad=True)
        nn.init.xavier_normal_(self.W, gain=1)

    def forward(self, x, label):
        n_examples = x.size()[0]
        n_pick = int(n_examples * self.thresh)
        x_norm = torch.norm(x, 2, 1, True).clamp(min=1e-12).expand_as(x)
        x_norm = x / x_norm
        w_norm = torch.norm(self.W, 2, 0, True).clamp(min=1e-12).expand_as(self
            .W)
        w_norm = self.W / w_norm
        cos_th = torch.mm(x_norm, w_norm)
        s_cos_th = self.scale * cos_th
        loss = self.cross_entropy(s_cos_th, label)
        loss, _ = torch.sort(loss, descending=True)
        loss = torch.mean(loss[:n_pick])
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_feats': 4, 'n_classes': 4}]
