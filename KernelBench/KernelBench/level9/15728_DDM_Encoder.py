import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
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


class DDM_Encoder(torch.nn.Module):

    def __init__(self, obs_space, dim, context_input=False, context_dim=0):
        """
        architecture should be input, so that we can pass multiple jobs !
        """
        super(DDM_Encoder, self).__init__()
        if context_input:
            self.linear1 = nn.Linear(obs_space + context_dim, dim)
        else:
            self.linear1 = nn.Linear(obs_space, dim)
        self.linear2 = nn.Linear(dim, 32 * 3 * 3)
        self.fc = nn.Linear(32 * 3 * 3, dim)
        self.apply(weights_init)
        self.train()

    def forward(self, inputs):
        x = F.elu(self.linear1(inputs))
        x = F.elu(self.linear2(x))
        x = F.tanh(self.fc(x))
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'obs_space': 4, 'dim': 4}]
