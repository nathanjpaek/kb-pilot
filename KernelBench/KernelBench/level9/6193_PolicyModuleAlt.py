import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyModuleAlt(nn.Module):

    def __init__(self, input_dim, hid_dim, n_actions):
        super().__init__()
        self.fc_1 = nn.Linear(input_dim, hid_dim)
        self.fc_2 = nn.Linear(hid_dim, hid_dim)
        self.fc_a = nn.Linear(hid_dim, n_actions)
        self.fc_v = nn.Linear(hid_dim, 1)

    def forward(self, image):
        batch_size, *_ = image.shape
        image = image.reshape(batch_size, -1)
        hidden = F.relu(self.fc_1(image))
        hidden = F.relu(self.fc_2(hidden))
        action = self.fc_a(hidden)
        value = self.fc_v(hidden)
        return action, value


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hid_dim': 4, 'n_actions': 4}]
