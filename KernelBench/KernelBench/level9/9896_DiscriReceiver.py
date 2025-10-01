import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.distributions


class DiscriReceiver(nn.Module):

    def __init__(self, n_features, n_hidden):
        super(DiscriReceiver, self).__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)

    def forward(self, x, _input):
        embedded_input = self.fc1(_input).tanh()
        dots = torch.matmul(embedded_input, torch.unsqueeze(x, dim=-1))
        return dots.squeeze()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4, 'n_hidden': 4}]
