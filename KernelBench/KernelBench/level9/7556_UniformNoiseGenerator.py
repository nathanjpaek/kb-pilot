import torch
import torch.distributions
import torch.utils.data


class AdversarialNoiseGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        raise NotImplementedError()


class UniformNoiseGenerator(AdversarialNoiseGenerator):

    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return (self.max - self.min) * torch.rand_like(x) + self.min


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
