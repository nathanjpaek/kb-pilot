import torch
import torch.nn.functional as F


class Encoder(torch.nn.Module):
    """Documentation for Encoder

    """

    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.e1 = torch.nn.Linear(input_dim, hidden_dim)
        self.e2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.e3 = torch.nn.Linear(hidden_dim, latent_dim)
        self.e4 = torch.nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.leaky_relu(self.e1(x))
        x = F.leaky_relu(self.e2(x))
        mean = self.e3(x)
        log_variance = self.e4(x)
        return mean, log_variance


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4, 'latent_dim': 4}]
