import torch
import numpy as np
from torch.nn import Parameter


class netmodel(torch.nn.Module):

    def __init__(self):
        super(netmodel, self).__init__()
        self.w0 = Parameter(torch.Tensor(1))
        self.w1 = Parameter(torch.Tensor(1))
        self.w0.data.uniform_(-1, 1)
        self.w1.data.uniform_(-1, 1)

    def forward(self, inputs):
        y = torch.stack([100 * self.w0 * inputs[:, 0], 0.1 * self.w1 *
            inputs[:, 1]])
        y = torch.t(y)
        return y.contiguous()

    def extract_grad(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.grad.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.grad.data.numpy().flatten()
            count += sz
        return pvec.copy()

    def extract_parameters(self):
        tot_size = self.count_parameters()
        pvec = np.zeros(tot_size, np.float32)
        count = 0
        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            pvec[count:count + sz] = param.data.numpy().flatten()
            count += sz
        return pvec.copy()

    def inject_parameters(self, pvec):
        self.count_parameters()
        count = 0
        for param in self.parameters():
            sz = param.data.numpy().flatten().shape[0]
            raw = pvec[count:count + sz]
            reshaped = raw.reshape(param.data.numpy().shape)
            param.data = torch.from_numpy(reshaped)
            count += sz
        return pvec

    def count_parameters(self):
        count = 0
        for param in self.parameters():
            count += param.data.numpy().flatten().shape[0]
        return count


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
