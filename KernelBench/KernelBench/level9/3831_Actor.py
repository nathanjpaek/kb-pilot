import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3_sigma = nn.Linear(256, action_dim)
        self.l3_mean = nn.Linear(256, action_dim)
        self.max_action = max_action
        self.action_dim = action_dim
        self.state_dim = state_dim

    def forward(self, state, *args):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.max_action * torch.tanh(self.l3_mean(a))
        sigma = F.softplus(self.l3_sigma(a)) + 0.001
        normal = Normal(mean, sigma)
        action = normal.rsample().clamp(-self.max_action, self.max_action)
        return action

    def forward_all(self, state, *args):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.max_action * torch.tanh(self.l3_mean(a))
        sigma = F.softplus(self.l3_sigma(a)) + 0.001
        normal = Normal(mean, sigma)
        action = normal.rsample()
        log_prob = normal.log_prob(action).sum(1, keepdim=True)
        action = action.clamp(-self.max_action, self.max_action)
        return action, log_prob, mean, sigma

    def forward_all_sample(self, state, *args):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        mean = self.max_action * torch.tanh(self.l3_mean(a))
        sigma = F.softplus(self.l3_sigma(a)) + 0.001
        normal = Normal(mean, sigma)
        action1 = normal.rsample().clamp(-self.max_action, self.max_action)
        action2 = normal.rsample().clamp(-self.max_action, self.max_action)
        prob1 = normal.log_prob(action1).sum(1, keepdim=True)
        prob2 = normal.log_prob(action2).sum(1, keepdim=True)
        probm = normal.log_prob(mean).sum(1, keepdim=True)
        prob1 = torch.exp(prob1)
        prob2 = torch.exp(prob2)
        probm = torch.exp(probm)
        return action1, action2, mean, sigma, prob1, prob2, probm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4}]
