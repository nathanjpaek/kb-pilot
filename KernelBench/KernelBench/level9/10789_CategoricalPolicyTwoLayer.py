import torch
import torch.nn.functional as F
import torch.distributions as td
import torch.nn as nn


class PolicyNetwork(nn.Module):
    """Base class for stochastic policy networks."""

    def __init__(self):
        super().__init__()

    def forward(self, state):
        """Take state as input, then output the parameters of the policy."""
        raise NotImplementedError('forward not implemented.')

    def sample(self, state):
        """
        Sample an action based on the model parameters given the current state.
        """
        raise NotImplementedError('sample not implemented.')


class CategoricalPolicy(PolicyNetwork):
    """
    Base class for categorical policy.

    Desired network needs to be implemented.
    """

    def __init__(self, state_dim, num_actions):
        super().__init__()
        self.state_dim = state_dim
        self.num_actions = num_actions

    def sample(self, state, no_log_prob=False):
        probs = self.forward(state)
        dist = td.Categorical(probs)
        action = dist.sample(sample_shape=torch.tensor([1]))
        return action if no_log_prob else (action, dist.log_prob(action))


class CategoricalPolicyTwoLayer(CategoricalPolicy):
    """
    Categorical policy using a fully connected two-layer network with ReLU
    activation to generate the parameters of the categorical distribution.
    """

    def __init__(self, state_dim, num_actions, hidden_layer1_size=256,
        hidden_layer2_size=256, init_std=0.01):
        super().__init__(state_dim, num_actions)
        self.init_std = init_std
        self.linear1 = nn.Linear(state_dim, hidden_layer1_size)
        self.linear2 = nn.Linear(hidden_layer1_size, hidden_layer2_size)
        self.linear3 = nn.Linear(hidden_layer2_size, num_actions)
        nn.init.normal_(self.linear1.weight, std=self.init_std)
        nn.init.normal_(self.linear2.weight, std=self.init_std)
        nn.init.normal_(self.linear3.weight, std=self.init_std)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        output = F.relu(self.linear3(x))
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'num_actions': 4}]
