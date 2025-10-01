import torch
from torch import nn


class FCLayer(nn.Module):

    def __init__(self, input_dim, output_dim, dropout_rate=0.1, is_active=
        True, is_dropout=True, active_type='mish'):
        """
        FC-Layer, mostly last output of model
        args:
            input_dim: input dimension, 输入维度, eg. 768
            output_dim: output dimension, 输出维度, eg. 32
            dropout_rate: dropout rate, 随机失活, eg. 0.1
            is_dropout: use dropout or not, 是否使用随机失活dropout, eg. True
            is_active: use activation or not, 是否使用激活函数如tanh, eg. True
            active_type: type of activate function, 激活函数类型, eg. "tanh", "relu"
        Returns:
            Tensor of batch.
        """
        super(FCLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.is_dropout = is_dropout
        self.active_type = active_type
        self.is_active = is_active
        self.softmax = nn.Softmax(1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()
        self.gelu = nn.GELU()

    def forward(self, x):
        if self.is_dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.is_active:
            if self.active_type.upper() == 'MISH':
                x = x * torch.tanh(nn.functional.softplus(x))
            elif self.active_type.upper() == 'SWISH':
                x = x * torch.sigmoid(x)
            elif self.active_type.upper() == 'TANH':
                x = self.tanh(x)
            elif self.active_type.upper() == 'GELU':
                x = self.gelu(x)
            elif self.active_type.upper() == 'RELU':
                x = self.relu(x)
            else:
                x = self.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
