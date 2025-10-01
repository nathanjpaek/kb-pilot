import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCriticModel(nn.Module):

    def __init__(self, n_state, n_actions):
        super(ActorCriticModel, self).__init__()
        self.fc1 = nn.Linear(n_state, 16)
        self.action1 = nn.Linear(16, 16)
        self.action2 = nn.Linear(16, n_actions)
        self.value1 = nn.Linear(16, 16)
        self.value2 = nn.Linear(16, 1)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        action_x = F.relu(self.action1(x))
        action_probs = F.softmax(self.action2(action_x), dim=-1)
        value_x = F.relu(self.value1(x))
        state_values = self.value2(value_x)
        return action_probs, state_values


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_state': 4, 'n_actions': 4}]
