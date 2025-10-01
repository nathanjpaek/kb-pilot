import torch
from torch import nn
import torch.jit
import torch.nn.functional


class BinaryDiceLoss(nn.Module):

    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0
            ], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)
        num = torch.sum(torch.mul(predict, target), dim=1)
        den = torch.sum(predict, dim=1) + torch.sum(target, dim=1
            ) + self.smooth
        dice_score = 2 * num / den
        loss_avg = 1 - dice_score.mean()
        return loss_avg


class DiceLoss4BraTS(nn.Module):

    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss4BraTS, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict %s & target %s shape do not match' % (
            predict.shape, target.shape)
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = torch.sigmoid(predict)
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1
                        ], 'Expect weight shape [{}], get[{}]'.format(target
                        .shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss
        return total_loss / (target.shape[1] - 1 if self.ignore_index is not
            None else target.shape[1])


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
