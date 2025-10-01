import math
import torch
from torch import nn


class LocallyConnected(nn.Module):
    """
    Local linear layer, i.e. Conv1dLocal() with filter size 1.
    """

    def __init__(self, num_linear: 'int', input_features: 'int',
        output_features: 'int', bias: 'bool'=True):
        """
        Create local linear layers.
        Transformations of the feature are independent of each other,
        each feature is expanded to several hidden units.

        Args:
            num_linear: num of local linear layers.
            input_features: m1.
            output_features: m2.
            bias: whether to include bias or not.
        """
        super().__init__()
        self.num_linear = num_linear
        self.input_features = input_features
        self.output_features = output_features
        self.weight = nn.Parameter(torch.Tensor(num_linear, input_features,
            output_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """
        Reset parameters
        """
        k = 1.0 / self.input_features
        bound = math.sqrt(k)
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        """
        Forward output calculation # [n, d, 1, m2] = [n, d, 1, m1] @ [1, d, m1, m2]

        Args:
            x: torch tensor

        Returns:
            output calculation
        """
        out = torch.matmul(x.unsqueeze(dim=2), self.weight.unsqueeze(dim=0))
        out = out.squeeze(dim=2)
        if self.bias is not None:
            out += self.bias
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_linear': 4, 'input_features': 4, 'output_features': 4}]
