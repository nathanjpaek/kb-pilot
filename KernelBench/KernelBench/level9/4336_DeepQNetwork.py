import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DeepQNetwork(nn.Module):

    def __init__(self, ALPHA):
        super(DeepQNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 19 * 8, 512)
        self.fc2 = nn.Linear(512, 6)
        self.optimizer = optim.RMSprop(self.parameters(), lr=ALPHA)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self

    def forward(self, observation):
        observation = T.Tensor(observation)
        observation = observation.view(-1, 1, 185, 95)
        observation = F.relu(self.conv1(observation))
        observation = F.relu(self.conv2(observation))
        observation = F.relu(self.conv3(observation))
        observation = observation.view(-1, 128 * 19 * 8)
        observation = F.relu(self.fc1(observation))
        actions = self.fc2(observation)
        return actions


def get_inputs():
    return [torch.rand([4, 1, 185, 95])]


def get_init_inputs():
    return [[], {'ALPHA': 4}]
