import torch
import torch.utils.data
import torch
import torch.nn as nn
from torch.autograd import Variable


class GaussianVAE2D(nn.Module):

    def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
        super(GaussianVAE2D, self).__init__()
        self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.en_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
        self.softplus = nn.Softplus()
        self.reset_parameters()

    def reset_parameters(self):
        self.en_mu.weight.data.normal_(0, 0.002)
        self.en_mu.bias.data.normal_(0, 0.002)
        self.en_sigma.weight.data.normal_(0, 0.002)
        self.en_sigma.bias.data.normal_(0, 0.002)

    def forward(self, x):
        mu = self.en_mu(x)
        sd = self.softplus(self.en_sigma(x))
        return mu, sd

    def sample(self, x):
        mu = self.en_mu(x)
        sd = self.softplus(self.en_sigma(x))
        noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu
            .size(3)))
        return mu + sd.mul(noise), mu, sd


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_in': 4, 'n_out': 4, 'kernel_size': 4, 'stride': 1}]
