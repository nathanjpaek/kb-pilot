import torch
import torch.nn as nn
import torch.utils.data


class JaccardLoss(nn.Module):
    """Jaccard loss.
    """

    def __init__(self, size_average=True, reduce=True, smooth=1.0):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth
        self.reduce = reduce

    def jaccard_loss(self, pred, target):
        loss = 0.0
        for index in range(pred.size()[0]):
            iflat = pred[index].view(-1)
            tflat = target[index].view(-1)
            intersection = (iflat * tflat).sum()
            loss += 1 - (intersection + self.smooth) / (iflat.sum() + tflat
                .sum() - intersection + self.smooth)
        return loss / float(pred.size()[0])

    def jaccard_loss_batch(self, pred, target):
        iflat = pred.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        loss = 1 - (intersection + self.smooth) / (iflat.sum() + tflat.sum(
            ) - intersection + self.smooth)
        return loss

    def forward(self, pred, target):
        if not target.size() == pred.size():
            raise ValueError(
                'Target size ({}) must be the same as pred size ({})'.
                format(target.size(), pred.size()))
        if self.reduce:
            loss = self.jaccard_loss(pred, target)
        else:
            loss = self.jaccard_loss_batch(pred, target)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
