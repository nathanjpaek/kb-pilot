import torch
import torch.nn.functional as F
from torch import nn


class OfflineTripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin=0.1):
        super(OfflineTripletLoss, self).__init__()
        self.margin = margin

    def forward(self, inputs, size_average=True):
        batchsize = inputs[0].size(0)
        anchor = inputs[0][0:int(batchsize / 3)]
        positive = inputs[0][int(batchsize / 3):int(batchsize * 2 / 3)]
        negative = inputs[0][int(batchsize * 2 / 3):]
        anchor = anchor.view(int(batchsize / 3), -1)
        positive = positive.view(int(batchsize / 3), -1)
        negative = negative.view(int(batchsize / 3), -1)
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
