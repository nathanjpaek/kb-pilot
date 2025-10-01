import torch
from torch import nn
import torch.nn.functional as F


class LearnableInterpolation(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int', trainable:
        'bool'=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = trainable
        self.lin = nn.Linear(in_features=self.input_size, out_features=self
            .output_size)
        self.init_weights()
        if self.trainable:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=
                torch.float32))
            self.gate = nn.Sigmoid()

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' not in name:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def forward(self, x):
        upsampled = self.lin(x)
        if self.trainable:
            upsampled = upsampled * self.gate(self.mask.unsqueeze(0)) * 2.0
        return upsampled


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


class ResampleNorm(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int'=None,
        trainable_add: 'bool'=True, residual_upsampling: 'str'=
        'interpolation', drop_normalization: 'bool'=False):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size
        self.residual_upsampling = residual_upsampling
        self.drop_normalization = drop_normalization
        if self.input_size != self.output_size:
            if self.residual_upsampling == 'interpolation':
                self.resample = TimeDistributedInterpolation(self.
                    output_size, batch_first=True, trainable=False)
            elif self.residual_upsampling == 'learnable':
                self.resample = LearnableInterpolation(self.input_size,
                    self.output_size, trainable=False)
            else:
                raise RuntimeError(
                    f'Wrong residual_upsampling method: {self.residual_upsampling}'
                    )
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=
                torch.float))
            self.gate = nn.Sigmoid()
        if self.drop_normalization is False:
            self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)
        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0
        if self.drop_normalization is False:
            output = self.norm(x)
        else:
            output = x
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4}]
