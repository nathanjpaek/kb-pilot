import torch
import torch.nn as nn
import torch.nn.functional as F


class mIoULoss(nn.Module):

    def __init__(self, weight=None, size_average=True, n_classes=4):
        super(mIoULoss, self).__init__()
        self.classes = n_classes

    def forward(self, inputs, target_oneHot):
        """
        IoU Loss for individual examples
        inputs - N x {Classes or higher} x H x W
        target_oneHot - N x {Classes or higher} x H x W
        BG can be ignored
        """
        N = inputs.size()[0]
        C = inputs.size()[1]
        inputs = F.softmax(inputs, dim=1)
        inter = inputs * target_oneHot
        inter = inter.view(N, C, -1).sum(2)
        union = inputs + target_oneHot - inputs * target_oneHot
        union = union.view(N, C, -1).sum(2)
        loss = inter / union
        return -(loss[:, -self.classes].mean() - 1.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
