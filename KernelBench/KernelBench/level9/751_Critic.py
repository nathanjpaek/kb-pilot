import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_encoder(nn.Module):

    def __init__(self):
        super(CNN_encoder, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(4, 8, kernel_size=3, padding=1,
            stride=1), nn.ReLU(), nn.MaxPool2d(4, 2), nn.Conv2d(8, 8,
            kernel_size=3, padding=1, stride=1), nn.ReLU(), nn.MaxPool2d(4,
            2), nn.Flatten())

    def forward(self, view_state):
        x = self.net(view_state)
        return x


class Critic(nn.Module):

    def __init__(self, state_space, hidden_size=64, cnn=False):
        super(Critic, self).__init__()
        self.is_cnn = cnn
        if self.is_cnn:
            self.encoder = CNN_encoder()
        self.linear_in = nn.Linear(state_space, hidden_size)
        self.state_value = nn.Linear(hidden_size, 1)

    def forward(self, x):
        if self.is_cnn:
            x = self.encoder(x)
        x = F.relu(self.linear_in(x))
        value = self.state_value(x)
        return value


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_space': 4}]
