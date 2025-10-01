import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        logprobs = F.log_softmax(logits, dim=-1)
        loss = -labels * logprobs
        loss = loss.sum(-1)
        return loss.mean()


class DenseCrossEntropy_smooth(nn.Module):

    def __init__(self, num_classes=4, label_smoothing=0.05):
        super(DenseCrossEntropy_smooth, self).__init__()
        self.smoothing = label_smoothing
        self.criterion = DenseCrossEntropy()
        self.num_classes = num_classes

    def forward(self, x, target):
        x = x.float()
        target.float()
        assert x.size(1) == self.num_classes
        target_smooth = (1 - self.smoothing
            ) * target + self.smoothing / self.num_classes
        return self.criterion(x, target_smooth)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
