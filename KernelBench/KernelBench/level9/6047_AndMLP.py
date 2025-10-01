import torch
import torch.nn as nn
import torch.nn.functional as F


class AndMLP(nn.Module):

    def __init__(self, n_layers, entity_dim):
        super(AndMLP, self).__init__()
        self.n_layers = n_layers
        self.layers = []
        for i in range(1, self.n_layers + 1):
            setattr(self, 'and_layer_{}'.format(i), nn.Linear(2 *
                entity_dim, 2 * entity_dim))
        self.last_layer = nn.Linear(2 * entity_dim, entity_dim)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=-1)
        for i in range(1, self.n_layers + 1):
            x = F.relu(getattr(self, 'and_layer_{}'.format(i))(x))
        x = self.last_layer(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_layers': 1, 'entity_dim': 4}]
