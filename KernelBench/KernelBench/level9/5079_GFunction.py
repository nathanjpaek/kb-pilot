import torch
import torch.nn.functional as F
from torch import nn
from torch import optim


class GFunction(nn.Module):

    def __init__(self, obs_size, num_outputs=128):
        super().__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs
        self.fc1 = nn.Linear(obs_size, 32)
        self.fc2 = nn.Linear(32, 32)
        self.last = nn.Linear(32, num_outputs)
        self.optimizer = optim.Adam(self.parameters(), lr=1e-05)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.last(x)
        return x

    def train_model(self, c_out, next_state):
        loss = nn.MSELoss()(c_out, self.forward(next_state))
        loss.backward()
        self.optimizer.step()
        return loss.item()


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_size': 4}]
