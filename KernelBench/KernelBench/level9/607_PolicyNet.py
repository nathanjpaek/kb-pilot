import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim
import torch.autograd
from torch.distributions import Normal


class PolicyNet(nn.Module):

    def __init__(self, learning_rate, lr_alpha, init_alpha, target_entropy,
        in_dim):
        self.target_entropy = target_entropy
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc_mu = nn.Linear(128, 1)
        self.fc_std = nn.Linear(128, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(init_alpha))
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action)
        real_action = torch.tanh(action)
        real_log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) +
            1e-07)
        return real_action, real_log_prob

    def train_net(self, q1, q2, mini_batch):
        s, _, _, _, _ = mini_batch
        a, log_prob = self.forward(s)
        entropy = -self.log_alpha.exp() * log_prob
        q1_val, q2_val = q1(s, a), q2(s, a)
        q1_q2 = torch.cat([q1_val, q2_val], dim=1)
        min_q = torch.min(q1_q2, 1, keepdim=True)[0]
        loss = -min_q - entropy
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.
            target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'learning_rate': 4, 'lr_alpha': 4, 'init_alpha': 4,
        'target_entropy': 4, 'in_dim': 4}]
