import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-05):
        super(PolicyNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        """
        Get output of the policy

        Args:
            state (torch array): State of the dynamical system
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.sigmoid_mod(self.linear3(x))
        return x

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.forward(state)
        action_np = action.detach().cpu().numpy()[0]
        return action_np

    def tanh_mod(self, x, p=1):
        x = x.float()
        x = 2 / (1 + torch.exp(-2 * (x / 100))) - 1
        x = x * p
        return x

    def sigmoid_mod(self, x, p=1.5):
        x = x.float()
        x = (2 / (1 + torch.exp(x) * 1) - 1) * -1
        x = x * p
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_inputs': 4, 'num_actions': 4, 'hidden_size': 4}]
