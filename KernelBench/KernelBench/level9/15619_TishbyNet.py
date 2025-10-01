import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)
    return t_log, running_mean


class ConcatLayer(nn.Module):

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x, y):
        return torch.cat((x, y), self.dim)


class CustomSequential(nn.Sequential):

    def forward(self, *input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)
        return input


class EMALoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()
        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / (running_mean + EPS
            ) / input.shape[0]
        return grad, None


class Mine(nn.Module):

    def __init__(self, T, loss='mine', alpha=0.01, method=None):
        super().__init__()
        self.running_mean = 0
        self.loss = loss
        self.alpha = alpha
        self.method = method
        if method == 'concat':
            if isinstance(T, nn.Sequential):
                self.T = CustomSequential(ConcatLayer(), *T)
            else:
                self.T = CustomSequential(ConcatLayer(), T)
        else:
            self.T = T

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]
        t = self.T(x, z).mean()
        t_marg = self.T(x, z_marg)
        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(t_marg, self.
                running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(t_marg, 0) - math.log(t_marg.shape[0]
                )
        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()
        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi

    def optimize(self, X, Y, iters, batch_size, opt=None):
        if opt is None:
            opt = torch.optim.Adam(self.parameters(), lr=0.0001)
        for iter in range(1, iters + 1):
            mu_mi = 0
            for x, y in utils.batch(X, Y, batch_size):
                opt.zero_grad()
                loss = self.forward(x, y)
                loss.backward()
                opt.step()
                mu_mi -= loss.item()
            if iter % (iters // 3) == 0:
                pass
        final_mi = self.mi(X, Y)
        None
        return final_mi


class TishbyNet(nn.Module):

    def __init__(self, input_dim, output_dim, activation='tanh', device='cpu'):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, 12)
        self.fc2 = nn.Linear(12, 10)
        self.fc3 = nn.Linear(10, 7)
        self.fc4 = nn.Linear(7, 5)
        self.fc5 = nn.Linear(5, 4)
        self.fc6 = nn.Linear(4, 3)
        self.fc7 = nn.Linear(3, output_dim)
        self.activation = activation
        self.softmax = nn.Softmax()

    def non_linear(self, x):
        if self.activation == 'tanh':
            return torch.tanh(x)
        elif self.activation == 'relu':
            return F.relu(x)
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.get_layer_outputs(x)[-1]

    def get_layer_outputs(self, x):
        x1 = self.non_linear(self.fc1(x))
        x2 = self.non_linear(self.fc2(x1))
        x3 = self.non_linear(self.fc3(x2))
        x4 = self.non_linear(self.fc4(x3))
        x5 = self.non_linear(self.fc5(x4))
        x6 = self.non_linear(self.fc6(x5))
        out = self.fc7(x6)
        return [x1, x2, x3, x4, x5, x6, out]

    def estimate_layerwise_mutual_information(self, x, target, iters):
        n, input_dim = target.shape
        layer_outputs = self.get_layer_outputs(x)
        layer_outputs[-1] = F.softmax(layer_outputs[-1])
        to_return = dict()
        for layer_id, layer_output in enumerate(layer_outputs):
            _, layer_dim = layer_output.shape
            statistics_network = nn.Sequential(nn.Linear(input_dim +
                layer_dim, 400), nn.ReLU(), nn.Linear(400, 400), nn.ReLU(),
                nn.Linear(400, 1))
            mi_estimator = Mine(T=statistics_network)
            mi = mi_estimator.optimize(target, layer_output.detach(), iters
                =iters, batch_size=n // 1, opt=None)
            to_return[layer_id] = mi.item()
        return to_return

    def calculate_information_plane(self, x, y, iters=100):
        info_x_t = self.estimate_layerwise_mutual_information(x, x, iters)
        info_y_t = self.estimate_layerwise_mutual_information(x, y, iters)
        return info_x_t, info_y_t


class T(nn.Module):

    def __init__(self, x_dim, z_dim):
        super().__init__()
        self.layers = CustomSequential(ConcatLayer(), nn.Linear(x_dim +
            z_dim, 400), nn.ReLU(), nn.Linear(400, 400), nn.ReLU(), nn.
            Linear(400, 400), nn.ReLU(), nn.Linear(400, 1))

    def forward(self, x, z):
        return self.layers(x, z)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
