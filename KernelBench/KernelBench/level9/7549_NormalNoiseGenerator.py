import torch
import torch.distributions
import torch.utils.data


class AdversarialNoiseGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        raise NotImplementedError()


class NormalNoiseGenerator(AdversarialNoiseGenerator):

    def __init__(self, sigma=1.0, mu=0):
        super().__init__()
        self.sigma = sigma
        self.mu = mu

    def forward(self, x):
        return self.sigma * torch.randn_like(x) + self.mu


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
