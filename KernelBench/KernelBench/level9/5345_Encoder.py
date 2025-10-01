import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Encoder, self).__init__()
        self.encoder_1 = nn.Linear(state_dim, 400)
        self.encoder_2 = nn.Linear(400, 300)
        self.encoder_3 = nn.Linear(300, 2 * action_dim)

    def forward(self, xu):
        encoded_out = F.relu(self.encoder_1(xu))
        encoded_out = F.relu(self.encoder_2(encoded_out))
        encoded_out = self.encoder_3(encoded_out)
        return encoded_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
