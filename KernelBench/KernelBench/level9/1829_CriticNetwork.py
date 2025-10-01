import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T


class CriticNetwork(nn.Module):

    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.action_value = nn.Linear(self.n_actions, self.fc1_dims)
        self.action_value2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)
        f1 = 1.0 / np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)
        f2 = 1.0 / np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)
        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)
        f4 = 1.0 / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)
        self.optimizer = optim.Adam(self.parameters(), lr=beta,
            weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self

    def forward(self, state, action):
        state_value = self.fc1(state)
        action_value = self.action_value(action)
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = F.relu(T.add(self.fc2(state_action_value),
            self.action_value2(state_action_value)))
        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self, chkpt_dir='tmp/ddpg'):
        None
        checkpoint_file = chkpt_dir + '/' + str(self.name + '_ddpg.pt')
        T.save(self.state_dict(), checkpoint_file)

    def load_checkpoint(self, chkpt_dir='tmp/ddpg'):
        None
        checkpoint_file = chkpt_dir + '/' + str(self.name + '_ddpg.pt')
        self.load_state_dict(T.load(checkpoint_file))

    def save_best(self, chkpt_dir='best/ddpg'):
        None
        checkpoint_file = chkpt_dir + '/' + str(self.name + '_ddpg.pt')
        T.save(self.state_dict(), checkpoint_file)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'beta': 4, 'input_dims': 4, 'fc1_dims': 4, 'fc2_dims': 4,
        'n_actions': 4, 'name': 4}]
