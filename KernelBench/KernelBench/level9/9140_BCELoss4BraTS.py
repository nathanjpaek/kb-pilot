import torch
from torch import nn
import torch.jit
import torch.nn.functional


class BCELoss4BraTS(nn.Module):

    def __init__(self, ignore_index=None, **kwargs):
        super(BCELoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.ignore_index = ignore_index
        self.criterion = nn.BCEWithLogitsLoss()

    def weighted_BCE_cross_entropy(self, output, target, weights=None):
        if weights is not None:
            assert len(weights) == 2
            output = torch.clamp(output, min=1e-07, max=1 - 1e-07)
            bce = weights[1] * (target * torch.log(output)) + weights[0] * ((
                1 - target) * torch.log(1 - output))
        else:
            output = torch.clamp(output, min=0.001, max=1 - 0.001)
            bce = target * torch.log(output) + (1 - target) * torch.log(1 -
                output)
        return torch.neg(torch.mean(bce))

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        total_loss = 0
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                bce_loss = self.criterion(predict[:, i], target[:, i])
                total_loss += bce_loss
        return total_loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
