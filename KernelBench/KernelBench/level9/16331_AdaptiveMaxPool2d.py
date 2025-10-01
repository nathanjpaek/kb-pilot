import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpikeAdaptiveMaxPoolNd(nn.Module):

    def __init__(self, output_size):
        super(_SpikeAdaptiveMaxPoolNd, self).__init__()
        self.output_size = output_size
        self.return_indices = True

    def reset_state(self):
        pass


class AdaptiveMaxPool2d(_SpikeAdaptiveMaxPoolNd):
    """Simple port of PyTorch AdaptiveMaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    The trace of the 'maximum' spike is also returned. In case of multiple spikes within pooling window, returns first spike of 
    the window (top left corner).
    """

    def forward(self, x, trace):
        x = x
        x, idx = F.adaptive_max_pool2d(x, self.output_size, self.return_indices
            )
        trace = trace[idx]
        return x > 0, trace


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([16, 4, 4, 4])]


def get_init_inputs():
    return [[], {'output_size': 4}]
