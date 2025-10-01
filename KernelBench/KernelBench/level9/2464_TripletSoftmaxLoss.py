import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletSoftmaxLoss(nn.Module):

    def __init__(self, margin=0.0, lambda_factor=0.01):
        super(TripletSoftmaxLoss, self).__init__()
        self.margin = margin
        self.loss_fn = nn.CrossEntropyLoss()
        self.lambda_factor = lambda_factor

    def forward(self, anchor, positive, negative, outputs, labels):
        distance_positive = torch.abs(anchor - positive).sum(1)
        distance_negative = torch.abs(anchor - negative).sum(1)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        loss_softmax = self.loss_fn(input=outputs, target=labels)
        loss_total = self.lambda_factor * losses.sum() + loss_softmax
        return loss_total, losses.sum(), loss_softmax


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
