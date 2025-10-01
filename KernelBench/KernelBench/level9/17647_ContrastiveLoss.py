import torch
from torchvision.transforms import functional as F
import torch.utils.data
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    output1/output2: embeddings nx2
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-09

    def forward(self, output1, output2, target, size_average=True):
        target = target
        distances = (output2 - output1).pow(2).sum(1)
        losses = 0.5 * (target * distances + (1 + -1 * target) * F.relu(
            self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'margin': 4}]
