import copy
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def square(a):
    return torch.pow(a, 2.0)


class Policy(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(num_inputs, 64)
        self.affine2 = nn.Linear(64, 64)
        self.action_mean = nn.Linear(64, num_outputs)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, num_outputs))
        self.module_list_current = [self.affine1, self.affine2, self.
            action_mean, self.action_log_std]
        self.module_list_old = [None] * len(self.module_list_current)
        self.backup()

    def backup(self):
        for i in range(len(self.module_list_current)):
            self.module_list_old[i] = copy.deepcopy(self.module_list_current[i]
                )

    def kl_div_p_q(self, p_mean, p_std, q_mean, q_std):
        """KL divergence D_{KL}[p(x)||q(x)] for a fully factorized Gaussian"""
        numerator = square(p_mean - q_mean) + square(p_std) - square(q_std)
        denominator = 2.0 * square(q_std) + eps
        return torch.sum(numerator / denominator + torch.log(q_std) - torch
            .log(p_std))

    def kl_old_new(self):
        """Gives kld from old params to new params"""
        kl_div = self.kl_div_p_q(self.module_list_old[-2], self.
            module_list_old[-1], self.action_mean, self.action_log_std)
        return kl_div

    def entropy(self):
        """Gives entropy of current defined prob dist"""
        ent = torch.sum(self.action_log_std + 0.5 * torch.log(2.0 * np.pi *
            np.e))
        return ent

    def forward(self, x, old=False):
        if old:
            x = F.tanh(self.module_list_old[0](x))
            x = F.tanh(self.module_list_old[1](x))
            action_mean = self.module_list_old[2](x)
            action_log_std = self.module_list_old[3].expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        else:
            x = F.tanh(self.affine1(x))
            x = F.tanh(self.affine2(x))
            action_mean = self.action_mean(x)
            action_log_std = self.action_log_std.expand_as(action_mean)
            action_std = torch.exp(action_log_std)
        return action_mean, action_log_std, action_std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_outputs': 4}]
