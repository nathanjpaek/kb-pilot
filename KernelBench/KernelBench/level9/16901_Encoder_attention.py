import torch
import torch.nn as nn


class Encoder_attention(nn.Module):

    def __init__(self, n_h):
        super(Encoder_attention, self).__init__()
        self.linear = nn.Linear(n_h, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """Output: X """
        x1 = self.linear(x).squeeze()
        weights = self.softmax(x1).unsqueeze(2)
        x2 = torch.sum(torch.mul(x, weights), dim=1)
        return x2, weights.squeeze().clone().detach()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_h': 4}]
