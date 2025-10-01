import torch
import torch.nn as nn
import torch.nn.functional as F


class _TimeDistributedInterpolation(nn.Module):

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


class _AddNorm(nn.Module):

    def __init__(self, input_size: 'int', skip_size: 'int'=None,
        trainable_add: 'bool'=True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.skip_size = skip_size or input_size
        if self.input_size != self.skip_size:
            self.resample = _TimeDistributedInterpolation(self.input_size,
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


class _GatedLinearUnit(nn.Module):
    """Gated Linear Unit"""

    def __init__(self, input_size: 'int', hidden_size: 'int'=None, dropout:
        'float'=None):
        super().__init__()
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = dropout
        self.hidden_size = hidden_size or input_size
        self.fc = nn.Linear(input_size, self.hidden_size * 2)
        self.init_weights()

    def init_weights(self):
        for n, p in self.named_parameters():
            if 'bias' in n:
                torch.nn.init.zeros_(p)
            elif 'fc' in n:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.fc(x)
        x = F.glu(x, dim=-1)
        return x


class _GateAddNorm(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int'=None,
        skip_size: 'int'=None, trainable_add: 'bool'=False, dropout:
        'float'=None):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size
        self.skip_size = skip_size or self.hidden_size
        self.dropout = dropout
        self.glu = _GatedLinearUnit(self.input_size, hidden_size=self.
            hidden_size, dropout=self.dropout)
        self.add_norm = _AddNorm(self.hidden_size, skip_size=self.skip_size,
            trainable_add=trainable_add)

    def forward(self, x, skip):
        output = self.glu(x)
        output = self.add_norm(output, skip)
        return output


class _ResampleNorm(nn.Module):

    def __init__(self, input_size: 'int', output_size: 'int'=None,
        trainable_add: 'bool'=True):
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add
        self.output_size = output_size or input_size
        if self.input_size != self.output_size:
            self.resample = _TimeDistributedInterpolation(self.output_size,
                batch_first=True, trainable=False)
        if self.trainable_add:
            self.mask = nn.Parameter(torch.zeros(self.output_size, dtype=
                torch.float))
            self.gate = nn.Sigmoid()
        self.norm = nn.LayerNorm(self.output_size)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        if self.input_size != self.output_size:
            x = self.resample(x)
        if self.trainable_add:
            x = x * self.gate(self.mask) * 2.0
        output = self.norm(x)
        return output


class _GatedResidualNetwork(nn.Module):

    def __init__(self, input_size: 'int', hidden_size: 'int', output_size:
        'int', dropout: 'float'=0.1, context_size: 'int'=None, residual:
        'bool'=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.residual = residual
        if self.input_size != self.output_size and not self.residual:
            residual_size = self.input_size
        else:
            residual_size = self.output_size
        if self.output_size != residual_size:
            self.resample_norm = _ResampleNorm(residual_size, self.output_size)
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.elu = nn.ELU()
        if self.context_size is not None:
            self.context = nn.Linear(self.context_size, self.hidden_size,
                bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.init_weights()
        self.gate_norm = _GateAddNorm(input_size=self.hidden_size,
            skip_size=self.output_size, hidden_size=self.output_size,
            dropout=self.dropout, trainable_add=False)

    def init_weights(self):
        for name, p in self.named_parameters():
            if 'bias' in name:
                torch.nn.init.zeros_(p)
            elif 'fc1' in name or 'fc2' in name:
                torch.nn.init.kaiming_normal_(p, a=0, mode='fan_in',
                    nonlinearity='leaky_relu')
            elif 'context' in name:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, context=None, residual=None):
        if residual is None:
            residual = x
        if self.input_size != self.output_size and not self.residual:
            residual = self.resample_norm(residual)
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x + context
        x = self.elu(x)
        x = self.fc2(x)
        x = self.gate_norm(x, residual)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4}]
