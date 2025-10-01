import torch
import torch.distributions
import torch.utils.data


class AdversarialNoiseGenerator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        return

    def forward(self, x):
        raise NotImplementedError()


class DeContraster(AdversarialNoiseGenerator):

    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        diff = torch.clamp(x.mean(dim=(1, 2, 3))[:, None, None, None] - x, 
            -self.eps, self.eps)
        return diff


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'eps': 4}]
