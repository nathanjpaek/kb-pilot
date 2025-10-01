import torch
from torch.distributions import Categorical
import torch.nn.functional as F
import torch.nn as nn


def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class CategoricalActor(nn.Module):

    def __init__(self, state_dim, hidden_dim, action_dim):
        super(CategoricalActor, self).__init__()
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        prob = F.softmax(x, -1)
        return prob

    def sample(self, state):
        prob = self.forward(state)
        distribution = Categorical(probs=prob)
        sample_action = distribution.sample().unsqueeze(-1)
        z = (prob == 0.0).float() * 1e-08
        logprob = torch.log(prob + z)
        greedy = torch.argmax(prob, dim=-1).unsqueeze(-1)
        return sample_action, prob, logprob, greedy


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'hidden_dim': 4, 'action_dim': 4}]
