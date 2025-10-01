import torch
import torch.nn as nn


class Loss(nn.Module):

    def __init__(self):
        super(Loss, self).__init__()
        self.BCELoss = nn.BCELoss(reduce=True, size_average=True)

    def forward(self, predict_y, input_y):
        loss = self.BCELoss(predict_y, input_y)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
