import torch
import torch as T
import torch.nn as nn
import torch.optim as optim


class CriticNetwork(nn.Module):

    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.ln1 = nn.LayerNorm(fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.ln2 = nn.LayerNorm(fc2_dim)
        self.q = nn.Linear(fc2_dim, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self

    def forward(self, state, action):
        x = T.cat([state, action], dim=-1)
        x = T.relu(self.ln1(self.fc1(x)))
        x = T.relu(self.ln2(self.fc2(x)))
        q = self.q(x)
        return q

    def save_checkpoint(self, checkpoint_file):
        T.save(self.state_dict(), checkpoint_file,
            _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(T.load(checkpoint_file))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'beta': 4, 'state_dim': 4, 'action_dim': 4, 'fc1_dim': 4,
        'fc2_dim': 4}]
