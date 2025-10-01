import torch
import torch.nn as nn
from collections import OrderedDict


class Decoder(nn.Module):

    def __init__(self, style_dim, class_dim):
        super(Decoder, self).__init__()
        self.linear_model = nn.Sequential(OrderedDict([('linear_1', nn.
            Linear(in_features=style_dim + class_dim, out_features=500,
            bias=True)), ('tan_h_1', nn.Tanh()), ('linear_2', nn.Linear(
            in_features=500, out_features=784, bias=True))]))

    def forward(self, style_latent_space, class_latent_space):
        x = torch.cat((style_latent_space, class_latent_space), dim=1)
        x = self.linear_model(x)
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'style_dim': 4, 'class_dim': 4}]
