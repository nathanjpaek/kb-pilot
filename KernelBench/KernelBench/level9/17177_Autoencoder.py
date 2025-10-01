import torch
from torch import nn


class Autoencoder(nn.Module):

    def __init__(self, input_dim, output_dim, n_hid, n_bottleneck):
        super(Autoencoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_bottleneck)
        self.fc3 = nn.Linear(n_bottleneck, n_hid)
        self.fc4 = nn.Linear(n_hid, output_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        None
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = self.act(x)
        y = self.fc4(x)
        return y

    def get_features(self, x):
        return None


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'n_hid': 4,
        'n_bottleneck': 4}]
