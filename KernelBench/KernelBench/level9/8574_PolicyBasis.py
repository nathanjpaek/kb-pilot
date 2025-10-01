import torch
import numpy as np
import torch.nn as nn


class PolicyBasis(nn.Module):

    def __init__(self, action_num, state_dim, task_dim):
        super(PolicyBasis, self).__init__()
        self.state_dim = state_dim
        self.task_dim = task_dim
        self.action_num = action_num
        self.weight_mu = nn.Parameter(torch.Tensor(action_num, state_dim,
            task_dim))
        self.policy_bias_mu = nn.Parameter(torch.Tensor(action_num))
        self.reward_bias_mu = nn.Parameter(torch.Tensor(action_num))
        self.reset_parameters()

    def forward(self, input1, input2, input3):
        N = input1.size(0)
        state_action_feat = torch.mm(input1, self.weight_mu.transpose(1, 0)
            .contiguous().view(self.state_dim, self.action_num * self.task_dim)
            ).view(N, self.action_num, self.task_dim)
        output1 = torch.bmm(state_action_feat, input2.unsqueeze(2)).squeeze(2)
        output2 = torch.bmm(state_action_feat, input3.unsqueeze(2)).squeeze(2)
        return (output1 + self.policy_bias_mu, output2 + self.
            reward_bias_mu, state_action_feat)

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.state_dim * self.task_dim * self.action_num
            )
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.policy_bias_mu.data.fill_(0)
        self.reward_bias_mu.data.fill_(0)

    def __repr__(self):
        return (self.__class__.__name__ +
            '(state_featurs={}, task_features={}, action_num={})'.format(
            self.state_dim, self.task_dim, self.action_num))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'action_num': 4, 'state_dim': 4, 'task_dim': 4}]
