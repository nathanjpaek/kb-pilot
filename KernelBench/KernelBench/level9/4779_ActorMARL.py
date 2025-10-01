import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorMARL(nn.Module):

    def __init__(self, dim_observation, dim_action):
        super(ActorMARL, self).__init__()
        self.FC1 = nn.Linear(dim_observation, 500)
        self.FC2 = nn.Linear(500, 128)
        self.FC3 = nn.Linear(128, dim_action)

    def forward(self, obs):
        result = F.relu(self.FC1(obs))
        result = F.relu(self.FC2(result))
        result = F.tanh(self.FC3(result))
        return result


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_observation': 4, 'dim_action': 4}]
