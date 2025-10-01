import torch
import torch.nn as nn


class weighted_mae_windows(nn.Module):

    def __init__(self, weights=(0.5, 1.2, 1.4, 1.6, 1.8, 2.0), thresholds=(
        5.0, 15.0, 30.0, 40.0, 45.0)):
        super(weighted_mae_windows, self).__init__()
        assert len(thresholds) + 1 == len(weights)
        self.weights = weights
        self.threholds = thresholds

    def forward(self, predict, target):
        """
        :param input: nbatchs * nlengths * nheigths * nwidths
        :param target: nbatchs * nlengths * nheigths * nwidths
        :return:
        """
        balance_weights = torch.zeros_like(target)
        balance_weights[target < self.threholds[0]] = self.weights[0]
        for i, _ in enumerate(self.threholds[:-1]):
            balance_weights[(target >= self.threholds[i]) & (target < self.
                threholds[i + 1])] = self.weights[i + 1]
        balance_weights[target >= self.threholds[-1]] = self.weights[-1]
        mae = torch.mean(balance_weights * torch.abs(predict - target))
        return mae


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
