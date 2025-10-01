import torch
import torch.utils.data


class LearnableTimeDepWeightedCost(torch.nn.Module):

    def __init__(self, time_horizon, dim=9, weights=None):
        super(LearnableTimeDepWeightedCost, self).__init__()
        if weights is None:
            self.weights = torch.nn.Parameter(0.01 * torch.ones([
                time_horizon, dim]))
        else:
            self.weights = weights
        self.clip = torch.nn.ReLU()
        self.dim = dim
        self.meta_grads = [[] for _, _ in enumerate(self.parameters())]

    def forward(self, y_in, y_target):
        assert y_in.dim() == 2
        mse = ((y_in[:, -self.dim:] - y_target[-self.dim:]) ** 2).squeeze()
        wmse = mse * self.weights
        return wmse.mean()


def get_inputs():
    return [torch.rand([4, 9]), torch.rand([4, 9])]


def get_init_inputs():
    return [[], {'time_horizon': 4}]
