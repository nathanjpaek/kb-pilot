import torch
import torch.nn as nn


class IOULoss(nn.Module):

    def __init__(self, eps: 'float'=1e-06):
        super(IOULoss, self).__init__()
        self.eps = eps

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0
            ], 'Predict and target must be same shape'
        dims = tuple(range(predict.ndimension())[1:])
        intersect = (predict * target).sum(dims) + self.eps
        union = (predict + target - predict * target).sum(dims) + self.eps
        return 1.0 - (intersect / union).sum() / intersect.nelement()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
