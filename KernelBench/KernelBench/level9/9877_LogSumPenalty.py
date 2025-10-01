from torch.nn import Module
import torch


class LogSumPenalty(Module):

    def __init__(self, epsilon=1):
        super(LogSumPenalty, self).__init__()
        self.epsilon = epsilon

    def forward(self, input):
        return torch.sum(torch.log(torch.abs(input) + self.epsilon))

    def eta_hat(self, w):
        w = torch.abs(w)
        return w * (w + self.epsilon)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
