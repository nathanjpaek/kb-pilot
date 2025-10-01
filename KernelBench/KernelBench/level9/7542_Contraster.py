import torch
import torch.distributions
import torch.utils.data


class AdversarialNoiseGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        raise NotImplementedError()


class Contraster(AdversarialNoiseGenerator):

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        eps = self.eps
        s = (x > 1 - eps).float() + torch.clamp(x * (x <= 1 - eps).float() -
            eps, 0, 1)
        return s - x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'eps': 4}]
