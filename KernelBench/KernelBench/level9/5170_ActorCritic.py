import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data


class ActorCritic(nn.Module):

    def __init__(self, num_states, num_actions, hidden_size):
        super(ActorCritic, self).__init__()
        self.num_actions = num_actions
        self.fc = nn.Linear(num_states, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        x = F.relu(self.fc(state))
        value = self.critic_linear2(x)
        policy_dist = F.softmax(self.actor_linear2(x))
        return value, policy_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_states': 4, 'num_actions': 4, 'hidden_size': 4}]
