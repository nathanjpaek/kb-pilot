import torch
import torch.nn as nn


class openai_critic(nn.Module):

    def __init__(self, obs_shape_n, action_shape_n):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, 128)
        self.linear_c2 = nn.Linear(128, 64)
        self.linear_c = nn.Linear(64, 1)
        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.
            calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.
            calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.
            calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input,
            action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'obs_shape_n': 4, 'action_shape_n': 4}]
