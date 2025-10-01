import torch
import torch.nn as nn


class EntityLayer(nn.Module):

    def __init__(self, initial_size, layer_size, device='cpu'):
        super(EntityLayer, self).__init__()
        self.weights_ent = nn.Linear(initial_size, layer_size, bias=False)
        self.init_params()
        self

    def init_params(self):
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)

    def forward(self, x, h):
        h_prime = self.weights_ent(x) + h
        return h_prime


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'initial_size': 4, 'layer_size': 1}]
