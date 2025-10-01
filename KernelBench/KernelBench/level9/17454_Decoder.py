import torch
from torch import nn


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, nlabels):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential()
        input_size = latent_size + nlabels
        for i, (in_size, out_size) in enumerate(zip([input_size] +
            layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(name='L%i' % i, module=nn.Linear(in_size,
                out_size))
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name='A%i' % i, module=nn.ReLU())
            else:
                self.MLP.add_module(name='softmax', module=nn.Softmax(dim=1))

    def forward(self, z, c):
        z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'layer_sizes': [4, 4], 'latent_size': 4, 'nlabels': 4}]
