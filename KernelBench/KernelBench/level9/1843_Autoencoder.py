from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import BCELoss


class Autoencoder(Module):

    def __init__(self, lr=0.002):
        super(Autoencoder, self).__init__()
        self.encode1 = Conv2d(1, 32, 3, padding=1)
        self.encode2 = Conv2d(32, 32, 3, padding=1)
        self.pool = MaxPool2d(2, 2)
        self.decode1 = Conv2d(32, 32, 3, padding=1)
        self.decode2 = Conv2d(32, 32, 3, padding=1)
        self.output = Conv2d(32, 1, 3, padding=1)
        self.loss = BCELoss()
        self.optimizer = optim.Adadelta(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu:0')
        self

    def forward(self, inputs):
        x = F.relu(self.encode1(inputs))
        x = self.pool(x)
        x = F.relu(self.encode2(x))
        x = self.pool(x)
        x = self.decode1(x)
        x = F.relu(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.decode2(x)
        x = F.relu(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = F.sigmoid(self.output(x))
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64])]


def get_init_inputs():
    return [[], {}]
