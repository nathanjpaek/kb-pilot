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


class VanillaNet(BasicNet):

    def predict(self, x, to_numpy=False):
        y = self.forward(x)
        if to_numpy:
            if type(y) is list:
                y = [y_.cpu().data.numpy() for y_ in y]
            else:
                y = y.cpu().data.numpy()
        return y


class FCNet(nn.Module, VanillaNet):

    def __init__(self, dims, optimizer_fn=None, gpu=True):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(dims[0], dims[1])
        self.fc2 = nn.Linear(dims[1], dims[2])
        self.fc3 = nn.Linear(dims[2], dims[3])
        BasicNet.__init__(self, optimizer_fn, gpu)

    def forward(self, x):
        x = self.to_torch_variable(x)
        x = x.view(x.size(0), -1)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return y


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dims': [4, 4, 4, 4]}]
