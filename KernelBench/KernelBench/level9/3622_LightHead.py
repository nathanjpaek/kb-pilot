import torch
from torch import nn


class RMSNorm(nn.Module):
    """An implementation of RMS Normalization.

    # https://catalyst-team.github.io/catalyst/_modules/catalyst/contrib/nn/modules/rms_norm.html#RMSNorm
    """

    def __init__(self, dimension: 'int', epsilon: 'float'=1e-08, is_bias:
        'bool'=False):
        """
        Args:
            dimension (int): the dimension of the layer output to normalize
            epsilon (float): an epsilon to prevent dividing by zero
                in case the layer has zero variance. (default = 1e-8)
            is_bias (bool): a boolean value whether to include bias term
                while normalization
        """
        super().__init__()
        self.dimension = dimension
        self.epsilon = epsilon
        self.is_bias = is_bias
        self.scale = nn.Parameter(torch.ones(self.dimension))
        if self.is_bias:
            self.bias = nn.Parameter(torch.zeros(self.dimension))

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        x_std = torch.sqrt(torch.mean(x ** 2, -1, keepdim=True))
        x_norm = x / (x_std + self.epsilon)
        if self.is_bias:
            return self.scale * x_norm + self.bias
        return self.scale * x_norm


class LightHead(nn.Module):

    def __init__(self, in_features, num_classes):
        super().__init__()
        self.norm = RMSNorm(in_features)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'num_classes': 4}]
