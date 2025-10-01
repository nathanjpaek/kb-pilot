import torch
import torch.nn as nn
from torch.nn import functional as F


class PointwiseConv(nn.Module):
    """
    Pointwise Convolution (1x1 Conv)

    Convolution 1 Dimension (Faster version)
    (cf. https://github.com/huggingface/pytorch-openai-transformer-lm/blob/        eafc28abdfadfa0732f03a0fc65805c5bfb2ffe7/model_pytorch.py#L45)

    * Args:
        input_size: the number of input tensor's dimension
        num_filters: the number of convolution filter
    """

    def __init__(self, input_size, num_filters):
        super(PointwiseConv, self).__init__()
        self.kernel_size = 1
        self.num_filters = num_filters
        weight = torch.empty(input_size, num_filters)
        nn.init.normal_(weight, std=0.02)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(num_filters))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.num_filters,)
        x = torch.addmm(self.bias, x.contiguous().view(-1, x.size(-1)),
            self.weight)
        x = x.view(*size_out)
        return x


class PositionwiseFeedForward(nn.Module):
    """
    Pointwise Feed-Forward Layer

    * Args:
        input_size: the number of input size
        hidden_size: the number of hidden size

    * Kwargs:
        dropout: the probability of dropout
    """

    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.pointwise_conv1 = PointwiseConv(input_size=input_size,
            num_filters=hidden_size)
        self.pointwise_conv2 = PointwiseConv(input_size=hidden_size,
            num_filters=input_size)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.pointwise_conv1(x)
        x = self.activation_fn(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
