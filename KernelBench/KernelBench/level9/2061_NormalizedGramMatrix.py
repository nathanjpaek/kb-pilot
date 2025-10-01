import torch
import torch.nn as nn


def normalize_by_stddev(tensor):
    """
    divides channel-wise by standard deviation of channel
    """
    channels = tensor.shape[1]
    stddev = tensor.std(dim=(0, 2)).view(1, channels, 1) + 1e-15
    return tensor.div(stddev)


class NormalizedGramMatrix(nn.Module):
    """
    I have found that normalizing the tensor before calculating the gram matrices leads to better convergence.
    """

    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        F = normalize_by_stddev(F)
        G = torch.bmm(F, F.transpose(1, 2))
        G = G.div_(h * w)
        return G


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
