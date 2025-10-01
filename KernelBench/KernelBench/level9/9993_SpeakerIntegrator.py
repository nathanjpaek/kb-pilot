import torch
import torch.nn as nn
import torch.utils.data


class SpeakerIntegrator(nn.Module):

    def __init__(self):
        super(SpeakerIntegrator, self).__init__()

    def forward(self, x, spembs):
        """
        x      shape : (batch, 39, 256)
        spembs shape : (batch, 256)
        """
        spembs = spembs.unsqueeze(1)
        spembs = spembs.repeat(1, x.shape[1], 1)
        x = x + spembs
        return x


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
