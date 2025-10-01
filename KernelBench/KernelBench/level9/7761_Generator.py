import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):

    def __init__(self, embed_size, max_size, nlayers=0, activation_type='tanh'
        ):
        super(Generator, self).__init__()
        hidden = max_size * embed_size
        if activation_type == 'tanh':
            activation = nn.Tanh()
        if nlayers > 0:
            hidden_layers = [nn.Linear(hidden, embed_size), activation]
            for n in range(1, nlayers):
                hidden_layers.append(nn.Linear(embed_size, embed_size))
                hidden_layers.append(activation)
            self.hidden_layers = nn.ModuleList(hidden_layers)
            self.proj = nn.Linear(embed_size, 2)
        else:
            self.hidden_layers = None
            self.proj = nn.Linear(hidden, 2)

    def forward(self, x):
        x = x.reshape(x.size()[0], -1)
        if self.hidden_layers is not None:
            for layer in self.hidden_layers:
                x = layer(x)
        x = self.proj(x)
        return F.log_softmax(x, dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'embed_size': 4, 'max_size': 4}]
