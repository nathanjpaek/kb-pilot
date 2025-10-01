import torch
import torch.nn as nn
import torch.nn.functional as func


class predicates1(nn.Module):

    def __init__(self, num_predicates, body_len):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(body_len, num_predicates).
            uniform_(0.0, 0.1))
        self.beta = nn.Parameter(torch.ones(1, body_len))

    def forward(self, x):
        beta, weights = self.get_params()
        ret = 1 - torch.clamp(-func.linear(x, weights) + beta, 0, 1)
        return ret

    def get_params(self):
        weights = func.relu(self.weights)
        beta = func.relu(self.beta)
        return beta, weights


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_predicates': 4, 'body_len': 4}]
