import torch
from torch import nn
import torch.nn.functional as fcnal


class softCrossEntropy(torch.nn.Module):

    def __init__(self, alpha=0.95):
        """
        :param alpha: Strength (0-1) of influence from soft labels in training
        """
        super(softCrossEntropy, self).__init__()
        self.alpha = alpha
        return

    def forward(self, inputs, target, true_labels):
        """
        :param inputs: predictions
        :param target: target (soft) labels
        :param true_labels: true (hard) labels
        :return: loss
        """
        KD_loss = self.alpha
        KD_loss *= nn.KLDivLoss(size_average=False)(fcnal.log_softmax(
            inputs, dim=1), fcnal.softmax(target, dim=1))
        KD_loss += (1 - self.alpha) * fcnal.cross_entropy(inputs, true_labels)
        return KD_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
