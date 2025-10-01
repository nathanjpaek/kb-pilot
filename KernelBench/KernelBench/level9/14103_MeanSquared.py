import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.parallel


def mean_squared(y, target, mask=None):
    y = y.softmax(1)
    loss = F.mse_loss(y, target, reduction='none').mean(1)
    if mask is not None:
        loss = mask * loss
    return loss.mean()


class MeanSquared(nn.Module):

    def __init__(self, use_onehot=False, num_classes=10):
        super(MeanSquared, self).__init__()
        self.use_onehot = use_onehot
        self.num_classes = num_classes

    def _make_one_hot(self, y):
        return torch.eye(self.num_classes)[y]

    def forward(self, y, target, mask=None, *args, **kwargs):
        if self.use_onehot:
            target = self._make_one_hot(target)
        return mean_squared(y, target.detach(), mask)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
