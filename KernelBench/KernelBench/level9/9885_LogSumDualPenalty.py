from torch.nn import Module
import torch


class LogSumDualPenalty(Module):

    def __init__(self, epsilon=1):
        super(LogSumDualPenalty, self).__init__()
        self.epsilon = epsilon

    def forward(self, input):
        eta = input
        sqrt = torch.sqrt(self.epsilon ** 2 + 4 * eta)
        return 2 * torch.sum(torch.log((sqrt + self.epsilon) / 2) - (sqrt -
            self.epsilon) ** 2 / (4 * eta))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
