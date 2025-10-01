import torch
import torch.nn as nn


class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs, discrete=False):
        super(Policy, self).__init__()
        self.discrete = discrete
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        if self.discrete:
            self.action_preds = nn.Softmax()
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.saved_actions = []
        self.rewards = []
        self.final_value = 0

    def forward(self, x):
        x = torch.tanh(self.affine1(x))
        x = torch.tanh(self.affine2(x))
        action_mean = self.action_mean(x)
        if self.discrete:
            action_mean = torch.sigmoid(action_mean)
            action_mean = self.action_preds(action_mean)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
