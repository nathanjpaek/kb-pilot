import torch
import torch.nn as nn


class ConvBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, momentum=0.9, epsilon=1e-05):
        """
        input: assume 4D input (mini_batch_size, # channel, w, h)
        momentum: momentum for exponential average
        """
        super(nn.BatchNorm2d, self).__init__(num_features)
        self.momentum = momentum
        self.insize = num_features
        self.epsilon = epsilon
        self.register_buffer('running_mean', torch.zeros(self.insize))
        self.register_buffer('running_var', torch.ones(self.insize))
        self.reset_parameters()

    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x, gamma=None, beta=None):
        if self.training is True:
            mean = x.mean([0, 2, 3])
            var = x.var([0, 2, 3])
            if self.track_running_stats is True:
                with torch.no_grad():
                    self.running_mean = self.momentum * self.running_mean + (
                        1.0 - self.momentum) * mean
                    self.running_var = self.momentum * self.running_var + (
                        1.0 - self.momentum) * (x.shape[0] / (x.shape[0] - 
                        1) * var)
        else:
            mean = self.running_mean
            var = self.running_var
        current_mean = mean.view([1, self.insize, 1, 1]).expand_as(x)
        current_var = var.view([1, self.insize, 1, 1]).expand_as(x)
        x = x - current_mean
        x = x / (current_var + self.eps) ** 0.5
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
