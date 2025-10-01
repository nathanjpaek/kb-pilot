import torch
import torch.nn as nn
from collections import OrderedDict


class Actor(nn.Module):

    def __init__(self, state_size, action_size, actor_fc_sizes=[256, 128, 64]):
        super(Actor, self).__init__()
        sequence_dict_actor = OrderedDict()
        sequence_dict_actor['fc0'] = nn.Linear(state_size, actor_fc_sizes[0])
        sequence_dict_actor['fc_rrelu0'] = nn.RReLU()
        for i, actor_fc_size in enumerate(actor_fc_sizes):
            if i == len(actor_fc_sizes) - 1:
                break
            sequence_dict_actor['fc{}'.format(i + 1)] = nn.Linear(actor_fc_size
                , actor_fc_sizes[i + 1])
            sequence_dict_actor['fc_rrelu{}'.format(i + 1)] = nn.RReLU()
        sequence_dict_actor['logit'] = nn.Linear(actor_fc_sizes[-1],
            action_size)
        self.fc_actor = nn.Sequential(sequence_dict_actor)
        self.tanh = nn.Tanh()

    def forward(self, common_res):
        actor_res = self.fc_actor(common_res)
        action = self.tanh(actor_res)
        return action


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
