from torch.nn import Module
import torch
from torch.nn.modules import Module
from torch.nn.modules import Linear


class MDN(Module):

    def __init__(self, input_size, num_mixtures):
        super(MDN, self).__init__()
        self.input_size = input_size
        self.num_mixtures = num_mixtures
        self.parameter_layer = Linear(in_features=input_size, out_features=
            1 + 6 * num_mixtures)

    def forward(self, input_, bias=None):
        mixture_parameters = self.parameter_layer(input_)
        eos_hat = mixture_parameters[:, :, 0:1]
        pi_hat, mu1_hat, mu2_hat, sigma1_hat, sigma2_hat, rho_hat = (torch.
            chunk(mixture_parameters[:, :, 1:], 6, dim=2))
        eos = torch.sigmoid(-eos_hat)
        mu1 = mu1_hat
        mu2 = mu2_hat
        rho = torch.tanh(rho_hat)
        if bias is None:
            bias = torch.zeros_like(rho)
        pi = torch.softmax(pi_hat * (1 + bias), dim=2)
        sigma1 = torch.exp(sigma1_hat - bias)
        sigma2 = torch.exp(sigma2_hat - bias)
        return eos, pi, mu1, mu2, sigma1, sigma2, rho

    def __repr__(self):
        s = '{name}(input_size={input_size}, num_mixtures={num_mixtures})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'num_mixtures': 4}]
