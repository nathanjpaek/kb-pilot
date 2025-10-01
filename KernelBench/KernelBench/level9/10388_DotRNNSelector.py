from _paritybench_helpers import _mock_config
import torch
import torch as th
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class DotRNNSelector(nn.Module):

    def __init__(self, input_shape, args):
        super(DotRNNSelector, self).__init__()
        self.args = args
        self.epsilon_start = self.args.epsilon_start
        self.epsilon_finish = self.args.role_epsilon_finish
        self.epsilon_anneal_time = self.args.epsilon_anneal_time
        self.epsilon_anneal_time_exp = self.args.epsilon_anneal_time_exp
        self.delta = (self.epsilon_start - self.epsilon_finish
            ) / self.epsilon_anneal_time
        self.role_action_spaces_update_start = (self.args.
            role_action_spaces_update_start)
        self.epsilon_start_t = 0
        self.epsilon_reset = True
        self.fc1 = nn.Linear(args.rnn_hidden_dim, 2 * args.rnn_hidden_dim)
        self.fc2 = nn.Linear(2 * args.rnn_hidden_dim, args.action_latent_dim)
        self.epsilon = 0.05

    def forward(self, inputs, role_latent):
        x = self.fc2(F.relu(self.fc1(inputs)))
        x = x.unsqueeze(-1)
        role_latent_reshaped = role_latent.unsqueeze(0).repeat(x.shape[0], 1, 1
            )
        role_q = th.bmm(role_latent_reshaped, x).squeeze()
        return role_q

    def select_role(self, role_qs, test_mode=False, t_env=None):
        self.epsilon = self.epsilon_schedule(t_env)
        if test_mode:
            self.epsilon = 0.0
        masked_q_values = role_qs.detach().clone()
        random_numbers = th.rand_like(role_qs[:, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_roles = Categorical(th.ones(role_qs.shape).float()).sample(
            ).long()
        picked_roles = pick_random * random_roles + (1 - pick_random
            ) * masked_q_values.max(dim=1)[1]
        return picked_roles

    def epsilon_schedule(self, t_env):
        if t_env is None:
            return 0.05
        if t_env > self.role_action_spaces_update_start and self.epsilon_reset:
            self.epsilon_reset = False
            self.epsilon_start_t = t_env
            self.epsilon_anneal_time = self.epsilon_anneal_time_exp
            self.delta = (self.epsilon_start - self.epsilon_finish
                ) / self.epsilon_anneal_time
        if t_env - self.epsilon_start_t > self.epsilon_anneal_time:
            epsilon = self.epsilon_finish
        else:
            epsilon = self.epsilon_start - (t_env - self.epsilon_start_t
                ) * self.delta
        return epsilon


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_shape': 4, 'args': _mock_config(epsilon_start=4,
        role_epsilon_finish=4, epsilon_anneal_time=4,
        epsilon_anneal_time_exp=4, role_action_spaces_update_start=4,
        rnn_hidden_dim=4, action_latent_dim=4)}]
