import torch
import torch.nn as nn


class ActivationNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.04, device='cuda', is_relative_detach=False):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.device = device

    def forward(self, x):
        if self.sigma > 0:
            with torch.no_grad():
                scale = self.sigma * x.detach(
                    ) if self.is_relative_detach else self.sigma * x
                sampled_noise = torch.ones_like(x).normal_() * scale
            x = x + sampled_noise
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
