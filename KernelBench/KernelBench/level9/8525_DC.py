import torch
from torch import nn
import torch.nn.functional


class DC(nn.Module):

    def __init__(self, nb_classes):
        super(DC, self).__init__()
        self.softmax = nn.Softmax(1)
        self.nb_classes = nb_classes

    @staticmethod
    def onehot(gt, shape):
        gt = gt.long()
        y_onehot = torch.zeros(shape)
        y_onehot = y_onehot
        y_onehot.scatter_(1, gt, 1)
        return y_onehot

    def reshape(self, output, target):
        output.shape[0]
        if not all([(i == j) for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)
        target = target.permute(0, 2, 3, 4, 1)
        output = output.permute(0, 2, 3, 4, 1)
        None
        return output, target

    def dice(self, output, target):
        output = self.softmax(output)
        if not all([(i == j) for i, j in zip(output.shape, target.shape)]):
            target = self.onehot(target, output.shape)
        sum_axis = list(range(2, len(target.shape)))
        s = 1e-19
        intersect = torch.sum(output * target, sum_axis)
        dice = 2 * intersect / (torch.sum(output, sum_axis) + torch.sum(
            target, sum_axis) + s)
        return 1.0 - dice.mean()

    def forward(self, output, target):
        result = self.dice(output, target)
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nb_classes': 4}]
