import torch
import torch.nn as nn


class DiceLoss(nn.Module):

    def __init__(self, size_average=True, ignore_index=-100, reduce=True):
        super(DiceLoss, self).__init__()
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.reduce = reduce
        self.softMax = nn.Softmax(dim=1)

    def forward(self, input, target, weight=None):
        self.weight = weight
        num_classes = input.size(1)
        input = self.softMax(input)
        one_hot = target.new(num_classes, num_classes).fill_(0)
        for i in range(num_classes):
            one_hot[i, i] = 1
        target_onehot = one_hot[target]
        target_onehot = target_onehot.unsqueeze(1).transpose(1, -1).squeeze(-1)
        target_onehot = target_onehot.float()
        input = input.float()
        loss = self.dice_loss(input, target_onehot, self.weight)
        return loss

    def dice_loss(self, input, target, weight=None):
        smooth = 1.0
        loss = 0.0
        n_classes = input.size(1)
        for c in range(n_classes):
            iflat = input[:, c].contiguous().view(-1)
            tflat = target[:, c].contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            if weight is not None:
                w = weight[c]
            else:
                w = 1 / n_classes
            w = torch.tensor(w)
            w = w.float()
            loss += w * (1 - (2.0 * intersection + smooth) / (iflat.sum() +
                tflat.sum() + smooth))
        return loss


def get_inputs():
    return [torch.rand([64, 4, 4, 4]), torch.ones([1024], dtype=torch.int64)]


def get_init_inputs():
    return [[], {}]
