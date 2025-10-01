from torch.nn import Module
import torch
from torch import Tensor


class Accuracy(Module):
    """
    Class for calculating the accuracy for a given prediction and the labels
    for comparison.
    Expects the inputs to be from a range of 0 to 1 and sets a crossing threshold at 0.5
    the labels are similarly rounded.
    """

    def forward(self, pred: 'Tensor', lab: 'Tensor') ->Tensor:
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :return: the calculated accuracy
        """
        return Accuracy.calculate(pred, lab)

    @staticmethod
    def calculate(pred: 'Tensor', lab: 'Tensor'):
        """
        :param pred: the models prediction to compare with
        :param lab: the labels for the data to compare to
        :return: the calculated accuracy
        """
        pred = pred >= 0.5
        lab = lab >= 0.5
        correct = (pred == lab).sum()
        total = lab.numel()
        acc = correct.float() / total * 100.0
        return acc


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
