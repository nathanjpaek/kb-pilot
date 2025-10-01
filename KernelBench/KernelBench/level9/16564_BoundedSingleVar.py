import torch


class BoundedSingleVar(torch.nn.Module):
    """Wrapper a single parameter to represent an unknown coefficient in inverse problem with the upper and lower bound.

    :param lower_bound: The lower bound for the parameter.
    :type lower_bound: float
    :param upper_bound: The upper bound for the parameter.
    :type upper_bound: float
    """

    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.value = torch.nn.Parameter(torch.Tensor([0.0]))
        self.layer = torch.nn.Sigmoid()
        self.ub, self.lb = upper_bound, lower_bound

    def forward(self, x) ->torch.Tensor:
        return x[:, :1] * 0.0 + self.layer(self.value) * (self.ub - self.lb
            ) + self.lb

    def get_value(self) ->torch.Tensor:
        return self.layer(self.value) * (self.ub - self.lb) + self.lb


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'lower_bound': 4, 'upper_bound': 4}]
