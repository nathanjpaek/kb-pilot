import torch
import torch.nn as nn
import torch.nn.functional as F


class _SpikeMaxPoolNd(nn.Module):

    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
        ceil_mode=False):
        super(_SpikeMaxPoolNd, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = True

    def reset_state(self):
        pass


class MaxPool2d(_SpikeMaxPoolNd):
    """Simple port of PyTorch MaxPool2d with small adjustment for spiking operations.
    
    Currently pooling only supports operations on floating point numbers, thus it casts the uint8 spikes to floats back and forth.
    The trace of the 'maximum' spike is also returned. In case of multiple spikes within pooling window, returns first spike of 
    the window (top left corner).
    """

    def forward(self, x, trace):
        x = x
        x, idx = F.max_pool2d(x, self.kernel_size, self.stride, self.
            padding, self.dilation, self.ceil_mode, self.return_indices)
        trace = trace.view(-1)[idx.view(-1)]
        trace = trace.view(idx.shape)
        return x > 0, trace


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'kernel_size': 4}]
