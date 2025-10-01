import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCriticNetwork, self).__init__()
        self.num_actions = num_actions
        self.critic_linear1 = nn.Linear(num_inputs, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.actor_linear1 = nn.Linear(num_inputs, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, num_actions)

    def forward(self, state_tensor):
        value = F.relu(self.critic_linear1(state_tensor))
        value = self.critic_linear2(value)
        policy_dist = F.relu(self.actor_linear1(state_tensor))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)
        return value, policy_dist


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4}]
