import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F


class TFSamepaddingLayer(nn.Module):
    """To align with tf `same` padding. 
    
    Putting this before any conv layer that need padding
    Assuming kernel has Height == Width for simplicity
    """

    def __init__(self, ksize, stride):
        super(TFSamepaddingLayer, self).__init__()
        self.ksize = ksize
        self.stride = stride

    def forward(self, x):
        if x.shape[2] % self.stride == 0:
            pad = max(self.ksize - self.stride, 0)
        else:
            pad = max(self.ksize - x.shape[2] % self.stride, 0)
        if pad % 2 == 0:
            pad_val = pad // 2
            padding = pad_val, pad_val, pad_val, pad_val
        else:
            pad_val_start = pad // 2
            pad_val_end = pad - pad_val_start
            padding = pad_val_start, pad_val_end, pad_val_start, pad_val_end
        x = F.pad(x, padding, 'constant', 0)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'ksize': 4, 'stride': 1}]
