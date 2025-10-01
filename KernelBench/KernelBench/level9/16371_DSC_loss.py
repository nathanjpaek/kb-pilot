import torch
import torch.nn as nn


class DSC_loss(nn.Module):

    def __init__(self):
        super(DSC_loss, self).__init__()
        self.epsilon = 1e-06
        return

    def forward(self, pred, target):
        batch_num = pred.shape[0]
        pred = pred.contiguous().view(batch_num, -1)
        target = target.contiguous().view(batch_num, -1)
        DSC = (2 * (pred * target).sum(1) + self.epsilon) / ((pred + target
            ).sum(1) + self.epsilon)
        return 1 - DSC.sum() / float(batch_num)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
