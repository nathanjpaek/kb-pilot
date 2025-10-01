import torch
import torch.utils.data
import torch.nn as nn


class LinearScalerModel(nn.Module):

    def __init__(self, load_from: 'dict'=None):
        super().__init__()
        initial = torch.zeros(4)
        initial[2] = 1
        initial[3] = 10
        self.params = nn.Parameter(initial, requires_grad=False)
        self.param_names = ['Min dose', 'Min density', 'Max dose',
            'Max density']
        if load_from is None:
            self.params.requires_grad = True
        else:
            self.params[0] = load_from['min_dose']
            self.params[1] = load_from['min_density']
            self.params[2] = load_from['max_dose']
            self.params[3] = load_from['max_density']

    def forward(self, x):
        x = x.clone()
        x[:, 0] -= self.params[0]
        x[:, 0] /= self.params[2] - self.params[0]
        if x.shape[1] == 2:
            x[:, 1] -= self.params[1]
            x[:, 1] /= self.params[3] - self.params[1]
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
