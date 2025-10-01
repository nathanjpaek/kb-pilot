import torch
import torch.nn.functional as F
import torch.nn as nn


class TimeDistributedInterpolation(nn.Module):

    def __init__(self, output_size: 'int', batch_first: 'bool'=False,
        trainable: 'bool'=False):
        super().__init__()
        self.output_size = output_size
        self.batch_first = batch_first
        self.trainable = trainable
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=
                torch.float32))
            self.gate = nn.Sigmoid()

    def interpolate(self, x):
        upsampled = F.interpolate(x.unsqueeze(1), self.output_size, mode=
            'linear', align_corners=True).squeeze(1)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.interpolate(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.interpolate(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class AddNorm(nn.Module):

    def __init__(self, input_size: 'int', skip_size: 'int'=None,
        trainable_add: 'bool'=True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        if self.input_size != self.skip_size:
            self.resample = TimeDistributedInterpolation(self.input_size,
                batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.input_size, dtype=
                torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.input_size)

    def forward(self, x: 'torch.Tensor', skip: 'torch.Tensor'):
        if self.input_size != self.skip_size:
            skip = self.resample(skip)
        if self.trainable_add:
            skip = skip * self.gate(self.mask) * 2.0
        output = self.norm(x + skip)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
