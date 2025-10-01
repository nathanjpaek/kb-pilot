import torch
import torch.nn as nn
import torch.optim
import torch.utils.data


class WeightedView(nn.Module):
    """Calculate weighted view
    
    Args:
        num_groups: int, number of groups (views)
        reduce_dimension: bool, default False. If True, reduce dimension dim
        dim: default -1. Only used when reduce_dimension is True
        
    Shape: 
        - Input: if dim is None, (N, num_features*num_groups)
        - Output: (N, num_features)
        
    Attributes:
        weight: (num_groups)
        
    Examples:
    
        >>> model = WeightedView(3)
        >>> x = Variable(torch.randn(1, 6))
        >>> print(model(x))
        >>> model = WeightedView(3, True, 1)
        >>> model(x.view(1,3,2))
    """

    def __init__(self, num_groups, reduce_dimension=False, dim=-1):
        super(WeightedView, self).__init__()
        self.num_groups = num_groups
        self.reduce_dimension = reduce_dimension
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(num_groups))
        self.weight.data.uniform_(-1.0 / num_groups, 1.0 / num_groups)

    def forward(self, x):
        self.normalized_weight = nn.functional.softmax(self.weight, dim=0)
        if self.reduce_dimension:
            assert x.size(self.dim) == self.num_groups
            dim = self.dim if self.dim >= 0 else self.dim + x.dim()
            if dim == x.dim() - 1:
                out = (x * self.weight).sum(-1)
            else:
                out = torch.transpose((x.transpose(dim, -1) * self.
                    normalized_weight).sum(-1), dim, -1)
        else:
            assert x.dim() == 2
            num_features = x.size(-1) // self.num_groups
            out = (x.view(-1, self.num_groups, num_features).transpose(1, -
                1) * self.normalized_weight).sum(-1)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'num_groups': 1}]
