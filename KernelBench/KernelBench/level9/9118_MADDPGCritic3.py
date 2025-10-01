import torch
from torch import nn


class MADDPGCritic3(nn.Module):
    """
	Critic which takes observation-action pairs of all agents and returns one q value for all
	"""

    def __init__(self, n_agents: 'int', act_dim: 'int', obs_dim: 'int',
        history: 'int'=0, hidden_dim: 'int'=32):
        super(MADDPGCritic3, self).__init__()
        in_features = n_agents * ((history + 1) * obs_dim + act_dim)
        self.linear1 = nn.Linear(in_features=in_features, out_features=
            hidden_dim)
        self.linear2 = nn.Linear(in_features=hidden_dim, out_features=
            hidden_dim)
        self.linear3 = nn.Linear(in_features=hidden_dim, out_features=1)
        self.activation = nn.ReLU()

    def forward(self, obs: 'torch.Tensor', act: 'torch.Tensor') ->torch.Tensor:
        """
		obs -> (batch_size, n_agents, history+1, obs_dim)
		act -> (batch_size, n_agents, act_dim) 
		"""
        x = torch.cat((torch.flatten(obs, start_dim=1), torch.flatten(act,
            start_dim=1)), dim=1)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_agents': 4, 'act_dim': 4, 'obs_dim': 4}]
