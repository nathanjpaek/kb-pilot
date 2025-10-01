from _paritybench_helpers import _mock_config
import torch
from torch.nn import functional as F
from torch import nn


class GymDqn(nn.Module):

    def __init__(self, args, action_space):
        super(GymDqn, self).__init__()
        self.atoms = args.atoms
        self.action_space = action_space
        self.input_size = args.history_length * args.state_dim
        self.fc_forward = nn.Linear(self.input_size, args.hidden_size)
        self.fc_h_v = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_h_a = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc_z_v = nn.Linear(args.hidden_size, action_space)
        self.fc_z_a = nn.Linear(args.hidden_size, action_space)

    def forward(self, x, log=False):
        x = self.extract(x)
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))
        q = v + a - a.mean(1, keepdim=True)
        return q

    def reset_noise(self):
        pass

    def extract(self, x):
        x = x.view(-1, self.input_size)
        x = self.fc_forward(x)
        x = F.relu(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(atoms=4, history_length=4, state_dim=
        4, hidden_size=4), 'action_space': 4}]
