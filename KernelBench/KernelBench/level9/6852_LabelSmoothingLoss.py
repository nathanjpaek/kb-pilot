import torch
import torch.nn.functional as F
import torch.nn as nn


class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.0):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def smooth_one_hot(self, target: 'torch.Tensor', classes: 'int',
        smoothing: 'float'=0.0):
        assert 0 <= smoothing < 1
        shape = target.size(0), classes
        with torch.no_grad():
            target = torch.empty(size=shape, device=target.device).fill_(
                smoothing / (classes - 1)).scatter_(1, target.data.
                unsqueeze(1), 1.0 - smoothing)
        return target

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor'):
        target = LabelSmoothingLoss.smooth_one_hot(self, target, input.size
            (-1), self.smoothing)
        lsm = F.log_softmax(input, -1)
        loss = -(target * lsm).sum(-1)
        loss = loss.mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
