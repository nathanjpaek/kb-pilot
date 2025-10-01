import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6.0 / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('GRUCell') != -1:
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


class baseRNN_predict(nn.Module):

    def __init__(self, h_size, obs_dim, num_actions, context_input=False):
        super(baseRNN_predict, self).__init__()
        self.l1 = nn.Linear(h_size, 64)
        self.l2 = nn.Linear(64, 128)
        self.l3 = nn.Linear(128, obs_dim)
        self.apply(weights_init)

    def forward(self, h):
        h = torch.relu(self.l1(h))
        h = torch.relu(self.l2(h))
        obs = self.l3(h)
        return obs


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'h_size': 4, 'obs_dim': 4, 'num_actions': 4}]
