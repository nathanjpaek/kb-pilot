import torch
import torch.nn as nn


class Attention(nn.Module):

    def __init__(self, dims, norm=False):
        super(Attention, self).__init__()
        self.norm = norm
        if self.norm:
            self.constrain = L2Constrain()
        else:
            self.transform = nn.Linear(dims, dims)
        self.context_vector = nn.Linear(dims, 1, bias=False)
        self.reset_parameters()

    def forward(self, x):
        if self.norm:
            weights = self.context_vector(x)
            weights = torch.add(torch.div(weights, 2.0), 0.5)
        else:
            x_tr = torch.tanh(self.transform(x))
            weights = self.context_vector(x_tr)
            weights = torch.sigmoid(weights)
        x = x * weights
        return x, weights

    def reset_parameters(self):
        if self.norm:
            nn.init.normal_(self.context_vector.weight)
            self.constrain(self.context_vector)
        else:
            nn.init.xavier_uniform_(self.context_vector.weight)
            nn.init.xavier_uniform_(self.transform.weight)
            nn.init.zeros_(self.transform.bias)

    def apply_contraint(self):
        if self.norm:
            self.constrain(self.context_vector)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dims': 4}]
