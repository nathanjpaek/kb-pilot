from torch.nn import Module
import torch
import torch.nn as nn
from torch.nn.modules import Module


class ContrastiveLoss(Module):

    def __init__(self, margin=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss = nn.BCELoss()

    def forward(self, output, label):
        label = label.view(label.size()[0])
        loss_same = label * torch.pow(output, 2)
        loss_diff = (1 - label) * torch.pow(torch.clamp(self.margin -
            output, min=0.0), 2)
        loss_contrastive = torch.mean(loss_same + loss_diff)
        return loss_contrastive


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4])]


def get_init_inputs():
    return [[], {}]
