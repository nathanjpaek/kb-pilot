import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillDiscriminator(nn.Module):
    """fully connected 200x200 layers for inferring q(z|s)"""

    def __init__(self, state_dim, nb_skills):
        super(SkillDiscriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim, 200)
        self.fc2 = nn.Linear(200, 200)
        self.out = nn.Linear(200, nb_skills)

    def forward(self, x):
        """return: scalar value"""
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        logits = self.out(x)
        return logits, nn.LogSoftmax(dim=1)(logits)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'nb_skills': 4}]
