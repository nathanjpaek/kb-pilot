import torch
from torch.nn.modules.loss import _Loss


class SSE(_Loss):
    """
    Definition: sum_squared_error = 1/2 * nn.MSELoss(reduction = 'sum')
    The backward is defined as: input-target
    """

    def __init__(self, under_penalty, over_penalty):
        super(SSE, self).__init__(under_penalty, over_penalty)
        self.under_penalty = under_penalty
        self.over_penalty = over_penalty

    def forward(self, input, target):
        res = (input - target) ** 2
        res[input < target] = res[input < target].mul(self.under_penalty)
        res[input > target] = res[input > target].mul(self.over_penalty)
        return res.sum() / 2


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'under_penalty': 4, 'over_penalty': 4}]
