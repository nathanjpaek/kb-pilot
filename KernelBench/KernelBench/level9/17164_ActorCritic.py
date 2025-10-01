import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """ Actor Critic neural network with shared body.
	The Actor maps states (actions) to action, log_probs, entropy.
	The Critic maps states to values.
	"""

    def __init__(self, state_size, action_size, seed=0):
        """ Initialize the neural net.
        
        Params
        ======
        	state_size: 	dimension of each input state
        	action_size: 	dimension of each output
        	seed: 			random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1_body = nn.Linear(state_size, 64)
        self.fc2_body = nn.Linear(64, 64)
        self.fc3_actor = nn.Linear(64, action_size)
        self.std = nn.Parameter(torch.ones(1, action_size))
        self.fc3_critic = nn.Linear(64, 1)

    def forward(self, state, action=None):
        x = torch.Tensor(state)
        x = F.relu(self.fc1_body(x))
        x = F.relu(self.fc2_body(x))
        mean = torch.tanh(self.fc3_actor(x))
        dist = torch.distributions.Normal(mean, F.softplus(self.std))
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        log_prob = torch.sum(log_prob, dim=1, keepdim=True)
        entropy = dist.entropy()
        entropy = torch.sum(entropy, dim=1, keepdim=True)
        value = self.fc3_critic(x)
        return action, log_prob, entropy, value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_size': 4, 'action_size': 4}]
