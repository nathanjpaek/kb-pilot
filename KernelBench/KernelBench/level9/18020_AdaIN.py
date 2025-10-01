import torch
import torch.optim


class AdaIN(torch.nn.Module):

    def __init__(self, epsilon: 'float'=1e-05):
        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def calc_vector_mean_std(self, x):
        std = torch.sqrt(torch.var(x, dim=1) + self.epsilon)
        mean = torch.mean(x, dim=1)
        return mean, std

    def forward(self, x: 'torch.Tensor', y: 'torch.Tensor') ->torch.Tensor:
        size = x.size()
        x_mean, x_std = self.calc_vector_mean_std(x)
        y_mean, y_std = self.calc_vector_mean_std(y)
        normalized = (x - x_mean.unsqueeze(-1).expand(size)) / x_std.unsqueeze(
            -1).expand(size)
        return normalized * y_std.unsqueeze(-1).expand(size
            ) + y_mean.unsqueeze(-1).expand(size)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
