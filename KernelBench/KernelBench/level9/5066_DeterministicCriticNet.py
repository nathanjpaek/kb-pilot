import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class BasicNet:

    def __init__(self, optimizer_fn, gpu, LSTM=False):
        self.gpu = gpu and torch.cuda.is_available()
        self.LSTM = LSTM
        if self.gpu:
            self
            self.FloatTensor = torch.FloatTensor
        else:
            self.FloatTensor = torch.FloatTensor

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        if self.gpu:
            x = x
        return Variable(x)

    def reset(self, terminal):
        if not self.LSTM:
            return
        if terminal:
            self.h.data.zero_()
            self.c.data.zero_()
        self.h = Variable(self.h.data)
        self.c = Variable(self.c.data)


class DeterministicCriticNet(nn.Module, BasicNet):

    def __init__(self, state_dim, action_dim, gpu=False, batch_norm=False,
        non_linear=F.relu, hidden_size=64):
        super(DeterministicCriticNet, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)
        self.non_linear = non_linear
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(hidden_size)
            self.bn2 = nn.BatchNorm1d(hidden_size)
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)
        self.init_weights()

    def init_weights(self):
        bound = 0.003
        self.layer3.weight.data.uniform_(-bound, bound)
        self.layer3.bias.data.fill_(0)

        def fanin(size):
            v = 1.0 / np.sqrt(size[1])
            return torch.FloatTensor(size).uniform_(-v, v)
        self.layer1.weight.data = fanin(self.layer1.weight.data.size())
        self.layer1.bias.data.fill_(0)
        self.layer2.weight.data = fanin(self.layer2.weight.data.size())
        self.layer2.bias.data.fill_(0)

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        action = self.to_torch_variable(action)
        x = self.non_linear(self.layer1(x))
        if self.batch_norm:
            x = self.bn1(x)
        x = self.non_linear(self.layer2(torch.cat([x, action], dim=1)))
        if self.batch_norm:
            x = self.bn2(x)
        x = self.layer3(x)
        return x

    def predict(self, x, action):
        return self.forward(x, action)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'state_dim': 4, 'action_dim': 4}]
