import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):

    def __init__(self, n_layersDecod, hidden_size, output_size=2):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(n_layersDecod * hidden_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.n_layersDecod = n_layersDecod
        self.hidden_size = hidden_size

    def forward(self, x):
        x = x.view(-1, self.n_layersDecod * self.hidden_size)
        x = F.relu(self.map1(x))
        x = F.relu(self.map2(x))
        return nn.Softmax()(self.map3(x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_layersDecod': 1, 'hidden_size': 4}]
