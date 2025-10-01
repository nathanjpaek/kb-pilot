import torch
import torch.utils.data
import torch.nn as nn
from torch.nn.parameter import Parameter


class InstanceNorm(nn.Module):

    def __init__(self, num_features, affine=True, eps=1e-05):
        """`num_features` number of feature channels
        """
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        self.scale = Parameter(torch.Tensor(num_features))
        self.shift = Parameter(torch.Tensor(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.scale.data.normal_(mean=0.0, std=0.02)
            self.shift.data.zero_()

    def forward(self, input):
        size = input.size()
        x_reshaped = input.view(size[0], size[1], size[2] * size[3])
        mean = x_reshaped.mean(2, keepdim=True)
        centered_x = x_reshaped - mean
        std = torch.rsqrt((centered_x ** 2).mean(2, keepdim=True) + self.eps)
        norm_features = (centered_x * std).view(*size)
        if self.affine:
            output = norm_features * self.scale[:, None, None] + self.shift[
                :, None, None]
        else:
            output = norm_features
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
