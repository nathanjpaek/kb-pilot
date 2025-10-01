import torch
import torch.nn as nn


class ClsCriterion(nn.Module):

    def __init__(self):
        super(ClsCriterion, self).__init__()

    def forward(self, predict, label, batch_weight=None):
        """
        :param predict: B*C log_softmax result
        :param label: B*C one-hot label
        :param batch_weight: B*1 0-1 weight for each item in a batch
        :return: cross entropy loss
        """
        if batch_weight is None:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1))
        else:
            cls_loss = -1 * torch.mean(torch.sum(predict * label, dim=1) *
                batch_weight)
        return cls_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
