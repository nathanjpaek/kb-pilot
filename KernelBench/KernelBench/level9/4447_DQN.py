from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.state_space = args.state_space
        self.fc1 = nn.Linear(self.state_space, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, args.action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return x

    def parameter_update(self, source):
        self.load_state_dict(source.state_dict())


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(state_space=4, hidden_size=4,
        action_space=4)}]
