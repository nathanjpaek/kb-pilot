import torch


class PixelNorm(torch.nn.Module):
    """
    PixelNorm from ProgressiveGAN
    """

    def forward(self, x):
        return x / (x.mean(dim=1, keepdim=True).sqrt() + 1e-08)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
