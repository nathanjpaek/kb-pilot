import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical


class ActorCritic(nn.Module):

    def __init__(self):
        super().__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

    def select_action(self, state, values, select_props):
        state = torch.from_numpy(state).float()
        props, value = self(Variable(state))
        dist = Categorical(props)
        action = dist.sample()
        log_props = dist.log_prob(action)
        values.append(value)
        select_props.append(log_props)
        return action.data[0]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
