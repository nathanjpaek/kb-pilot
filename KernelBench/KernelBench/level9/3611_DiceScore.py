import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.backends.cudnn
import torch.utils.data


class DiceScore(nn.Module):

    def __init__(self, threshold=0.5):
        super(DiceScore, self).__init__()
        self.threshold = threshold

    def forward(self, logits, labels):
        probs = F.sigmoid(logits)
        num = labels.size(0)
        predicts = (probs.view(num, -1) > self.threshold).float()
        labels = labels.view(num, -1)
        intersection = predicts * labels
        score = 2.0 * intersection.sum(1) / (predicts.sum(1) + labels.sum(1))
        return score.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
