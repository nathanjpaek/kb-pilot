from _paritybench_helpers import _mock_config
import torch
import torch as th
import torch.nn as nn


class DotRole(nn.Module):

    def __init__(self, args):
        super(DotRole, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.q_fc = nn.Linear(args.rnn_hidden_dim, args.action_latent_dim)
        self.action_space = th.ones(args.n_actions)

    def forward(self, h, action_latent):
        role_key = self.q_fc(h)
        role_key = role_key.unsqueeze(-1)
        action_latent_reshaped = action_latent.unsqueeze(0).repeat(role_key
            .shape[0], 1, 1)
        q = th.bmm(action_latent_reshaped, role_key).squeeze()
        return q

    def update_action_space(self, new_action_space):
        self.action_space = th.Tensor(new_action_space).float()


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(n_actions=4, rnn_hidden_dim=4,
        action_latent_dim=4)}]
