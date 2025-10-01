import torch
import torch.nn as nn


class AlphaEntropy(nn.Module):

    def __init__(self):
        super().__init__()
        self.v_loss = nn.MSELoss()

    def forward(self, props, v, pi, reward):
        v_loss = self.v_loss(v, reward)
        p_loss = -torch.mean(torch.sum(props * pi, 1))
        return p_loss + v_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
