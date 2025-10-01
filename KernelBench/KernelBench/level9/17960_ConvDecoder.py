import torch
import torch.nn as nn
from torch.nn import functional as F


class ConvDecoder(nn.Module):

    def __init__(self, hidden_size, state_size, embedding_size, act_fn='relu'):
        super().__init__()
        self.act_fn = getattr(F, act_fn)
        self.embedding_size = embedding_size
        self.fc_1 = nn.Linear(hidden_size + state_size, embedding_size)
        self.conv_1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2)
        self.conv_2 = nn.ConvTranspose2d(128, 64, 5, stride=2)
        self.conv_3 = nn.ConvTranspose2d(64, 32, 6, stride=2)
        self.conv_4 = nn.ConvTranspose2d(32, 3, 6, stride=2)

    def forward(self, hidden, state):
        out = self.fc_1(torch.cat([hidden, state], dim=1))
        out = out.view(-1, self.embedding_size, 1, 1)
        out = self.act_fn(self.conv_1(out))
        out = self.act_fn(self.conv_2(out))
        out = self.act_fn(self.conv_3(out))
        obs = self.conv_4(out)
        return obs


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'state_size': 4, 'embedding_size': 4}]
