import torch
from torch import nn


class MCDO(nn.Module):

    def __init__(self, in_dim, out_dim, n_layers=1, hid_dim=50, p=0.05):
        super().__init__()
        self.n_layers = n_layers
        self.linear_in = nn.Linear(in_dim, hid_dim)
        nn.init.normal_(self.linear_in.weight, std=1 / (4 * hid_dim) ** 0.5)
        nn.init.zeros_(self.linear_in.bias)
        self.dropout_in = nn.Dropout(p)
        if n_layers > 1:
            models = list(range(3 * (n_layers - 1)))
            for i in range(0, len(models), 3):
                models[i] = nn.Linear(hid_dim, hid_dim)
                nn.init.normal_(models[i].weight, std=1 / (4 * hid_dim) ** 0.5)
                nn.init.zeros_(models[i].bias)
            for i in range(1, len(models), 3):
                models[i] = nn.ReLU()
            for i in range(2, len(models), 3):
                models[i] = nn.Dropout(p)
            self.hid_layers = nn.Sequential(*models)
        self.linear_out = nn.Linear(hid_dim, out_dim)
        nn.init.normal_(self.linear_out.weight, std=1 / (4 * out_dim) ** 0.5)
        nn.init.zeros_(self.linear_out.bias)

    def forward(self, x):
        x = torch.relu(self.linear_in(x))
        x = self.dropout_in(x)
        if self.n_layers > 1:
            x = self.hid_layers(x)
        x = self.linear_out(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
