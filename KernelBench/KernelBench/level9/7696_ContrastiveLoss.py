import torch
from torch import nn
from torch.nn import functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function. ref: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-09

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2)
        losses = 0.5 * (label.float() * distance + (1 + -1 * label).float() *
            F.relu(self.margin - (distance + self.eps).sqrt()).pow(2))
        loss_contrastive = torch.mean(losses)
        return loss_contrastive


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
