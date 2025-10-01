import torch
import torch.nn.functional as F
import torch.nn as nn


class ActorSAC(nn.Module):

    def __init__(self, state_dim, hidden, min_log_std=-20, max_log_std=2):
        super(ActorSAC, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.mu_head = nn.Linear(hidden, 1)
        self.log_std_head = nn.Linear(hidden, 1)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.
            max_log_std)
        return mu, log_std_head


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'hidden': 4}]
