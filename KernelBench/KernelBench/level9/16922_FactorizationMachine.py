from torch.nn import Module
import torch
from torch import FloatTensor
from torch.nn import Parameter


class SecondOrderInteraction(Module):
    """
    Factorized parameters for the Second Order Interactions

    Parameters
    ----------
    n_features: int
        Length of the input vector.
    n_factors: int, optional
        Number of factors of the factorized parameters
    """

    def __init__(self, n_features, n_factors):
        super(SecondOrderInteraction, self).__init__()
        self.batch_size = None
        self.n_features = n_features
        self.n_factors = n_factors
        self.v = Parameter(torch.Tensor(self.n_features, self.n_factors))
        self.v.data.uniform_(-0.01, 0.01)

    def forward(self, x):
        self.batch_size = x.size()[0]
        pow_x = torch.pow(x, 2)
        pow_v = torch.pow(self.v, 2)
        pow_sum = torch.pow(torch.mm(x, self.v), 2)
        sum_pow = torch.mm(pow_x, pow_v)
        out = 0.5 * (pow_sum - sum_pow).sum(1)
        return out.unsqueeze(-1)


class FactorizationMachine(Module):
    """
    Pointwise Factorization Machine Model

    Parameters
    ----------
    n_features: int
        Length of the input vector.
    n_factors: int, optional
        Number of factors of the factorized parameters
    """

    def __init__(self, n_features, n_factors=10):
        super(FactorizationMachine, self).__init__()
        self.n_features, self.factors = n_features, n_factors
        self.global_bias = Parameter(FloatTensor(1))
        self.global_bias.data.uniform_(-0.01, 0.01)
        self.linear = Parameter(FloatTensor(self.n_features))
        self.linear.data.uniform_(-0.01, 0.01)
        self.second_order = SecondOrderInteraction(self.n_features, self.
            factors)

    @property
    def v(self):
        return self.second_order.v

    @v.getter
    def v(self):
        return self.second_order.v

    def forward(self, x):
        mm = x * self.linear
        linear = mm.sum(1).unsqueeze(-1)
        interaction = self.second_order(x)
        res = self.global_bias + linear + interaction
        return res.squeeze(-1)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_features': 4}]
