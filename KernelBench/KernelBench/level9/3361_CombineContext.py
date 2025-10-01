import torch
from torch import nn


class CombineContext(nn.Module):

    def __init__(self, num_features, num_context_features):
        super(CombineContext, self).__init__()
        self.linear = nn.Linear(num_features + num_context_features,
            num_features)

    def forward(self, token, prev_context_vector):
        x = torch.cat((token, prev_context_vector), 1)
        return self.linear(x)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4, 'num_context_features': 4}]
