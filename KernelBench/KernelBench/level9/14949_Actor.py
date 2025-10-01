from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Actor(nn.Module):

    def __init__(self, state_size, action_size, args, log_std_min=-20,
        log_std_max=2):
        super(Actor, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.fc1 = nn.Linear(state_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, action_size)
        self.fc4 = nn.Linear(args.hidden_size, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu = self.fc3(x)
        log_std = self.fc4(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.
            log_std_max)
        std = torch.exp(log_std)
        return mu, std


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4, 'args': _mock_config(
        hidden_size=4)}]
