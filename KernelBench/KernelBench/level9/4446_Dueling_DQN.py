from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dueling_DQN(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.state_space = args.state_space
        self.fc1 = nn.Linear(self.state_space, args.hidden_size)
        self.action_space = args.action_space
        self.fc_h = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_z_v = nn.Linear(args.hidden_size, 1)
        self.fc_z_a = nn.Linear(args.hidden_size, self.action_space)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc_h(x))
        v, a = self.fc_z_v(x), self.fc_z_a(x)
        a_mean = torch.stack(a.chunk(self.action_space, 1), 1).mean(1)
        x = v.repeat(1, self.action_space) + a - a_mean.repeat(1, self.
            action_space)
        return x

    def parameter_update(self, source):
        self.load_state_dict(source.state_dict())


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(state_space=4, hidden_size=4,
        action_space=4)}]
