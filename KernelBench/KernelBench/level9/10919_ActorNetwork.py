import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


class ActorNetwork(nn.Module):

    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, max_action,
        n_actions):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.reparam_noise = 1e-06
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu(x)
        sigma = self.sigma(x)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)
        return mu, sigma

    def sample_normal(self, x, reparameterize=True):
        mu, sigma = self.forward(x)
        probabilities = Normal(mu, sigma)
        if reparameterize:
            action = probabilities.rsample()
        else:
            action = probabilities.sample()
        bounded_action = T.tanh(action) * T.tensor(self.max_action)
        log_probs = probabilities.log_prob(action)
        log_probs -= T.log(1 - bounded_action.pow(2) + self.reparam_noise)
        log_probs_sum = log_probs.sum(1, keepdim=True)
        return bounded_action, log_probs_sum


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'alpha': 4, 'input_dims': 4, 'fc1_dims': 4, 'fc2_dims': 4,
        'max_action': 4, 'n_actions': 4}]
