import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor1D(nn.Module):

    def __init__(self, state_dim, action_dim, max_action, option_num=3):
        super(Actor1D, self).__init__()
        """
        Input size: (batch_num, channel = state_dim * option_num, length = 1)
        """
        self.conv1 = nn.Conv1d(state_dim * option_num, 400 * option_num,
            kernel_size=1, groups=option_num)
        self.conv2 = nn.Conv1d(400 * option_num, 300 * option_num,
            kernel_size=1, groups=option_num)
        self.conv3 = nn.Conv1d(300 * option_num, action_dim * option_num,
            kernel_size=1, groups=option_num)
        self.max_action = max_action
        self.option_num = option_num

    def forward(self, x):
        x = x.view(x.shape[0], -1, 1).repeat(1, self.option_num, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_action * torch.tanh(self.conv3(x))
        x = x.view(x.shape[0], self.option_num, -1)
        x = x.transpose(dim0=1, dim1=2)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4, 'max_action': 4}]
