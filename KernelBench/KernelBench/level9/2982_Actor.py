import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


class Actor(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(input_dim, output_dim, bias=True)

    def _format(self, state):
        x = state
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.unsqueeze(0)
        return x

    def forward(self, a, p):
        input = self._format(torch.cat((a, p), dim=1))
        output = self.layer1(input)
        output = torch.clamp(output, 0, 1)
        output = F.normalize(output, dim=0)
        return output

    def full_pass(self, a, p):
        logits = self.forward(a, p)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        logpa = dist.log_prob(action).unsqueeze(-1)
        entropy = dist.entropy().unsqueeze(-1)
        is_exploratory = action != np.argmax(logits.detach().numpy())
        return action.item(), is_exploratory.item(), logpa, entropy

    def select_action(self, a, p):
        logits = self.forward(a, p)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action.item()

    def select_greedy_action(self, a, p):
        logits = self.forward(a, p)
        return np.argmax(logits.detach().numpy())


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4}]
