import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.checkpoint


class ScalarAttention(nn.Module):

    def __init__(self, in_size, hidden_size):
        super(ScalarAttention, self).__init__()
        self.hidden = nn.Linear(in_size, hidden_size)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.out = nn.Linear(hidden_size, 1)
        nn.init.orthogonal_(self.hidden.weight.data)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        self.alpha = self.softmax(self.out(torch.tanh(self.hidden(input))))
        x = (self.alpha.expand_as(input) * input).sum(dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_size': 4, 'hidden_size': 4}]
