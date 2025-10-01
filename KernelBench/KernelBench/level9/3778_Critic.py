import torch
import torch.nn as nn
from collections import OrderedDict


class Critic(nn.Module):

    def __init__(self, state_size, action_size, critic_fc_sizes=[256, 128, 64]
        ):
        super(Critic, self).__init__()
        sequence_dict_critic = OrderedDict()
        self.critic_first_layer = nn.Linear(state_size, critic_fc_sizes[0])
        sequence_dict_critic['fc0'] = nn.Linear(critic_fc_sizes[0] +
            action_size, critic_fc_sizes[0])
        sequence_dict_critic['fc_rrelu0'] = nn.RReLU()
        for i, critic_fc_size in enumerate(critic_fc_sizes):
            if i == len(critic_fc_sizes) - 1:
                break
            sequence_dict_critic['fc{}'.format(i + 1)] = nn.Linear(
                critic_fc_size, critic_fc_sizes[i + 1])
            sequence_dict_critic['fc_rrelu{}'.format(i + 1)] = nn.RReLU()
        sequence_dict_critic['logit'] = nn.Linear(critic_fc_sizes[-1], 1)
        self.fc_critic = nn.Sequential(sequence_dict_critic)

    def forward(self, common_res, action):
        common_res = self.critic_first_layer(common_res)
        common_res = nn.RReLU(common_res)
        common_res = torch.cat((common_res.lower, action), dim=1)
        value = self.fc_critic(common_res)
        return value


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
