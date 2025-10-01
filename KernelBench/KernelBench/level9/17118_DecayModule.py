import math
import torch
import torch.nn as nn


class DecayModule(nn.Module):

    def __init__(self, input_size, hidden_size, bias=True, num_chunks=1,
        activation='relu', nodiag=False):
        super(DecayModule, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nodiag = nodiag
        self.bias = bias
        self.num_chunks = num_chunks
        self.rgate = nn.Parameter(torch.tensor(0.8), requires_grad=True)
        self.weight_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size,
            input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size,
            hidden_size))
        self.d_rec = nn.Parameter(torch.zeros(num_chunks * hidden_size,
            hidden_size), requires_grad=False)
        self.activation = activation
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(num_chunks * hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
        self.reset_parameters()
        if self.nodiag:
            for i in range(hidden_size):
                self.weight_hh.data[i, i] = 0

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)
        for name, param in self.named_parameters():
            if name == 'rgate':
                param.data = torch.tensor(1.4)
        for i in range(self.num_chunks):
            x = i * self.hidden_size
            for j in range(self.hidden_size):
                if j < 0.8 * self.hidden_size:
                    self.d_rec[x + j][j] = 1.0
                else:
                    self.d_rec[x + j][j] = -1.0

    def forward(self, input_, hx=None):
        if hx is None:
            hx = input_.new_zeros(self.num_chunks * self.hidden_size,
                requires_grad=False)
        dale_hh = torch.mm(self.relu(self.weight_hh), self.d_rec)
        if self.bias:
            w_x = self.bias_ih + torch.matmul(self.weight_ih, input_).t()
            w_h = self.bias_hh + torch.matmul(dale_hh, hx.t()).t()
        else:
            w_x = torch.matmul(self.weight_ih, input_).t()
            w_h = torch.matmul(dale_hh, hx.t()).t()
        w_w = self.sigmoid(self.rgate) * hx + (1 - self.sigmoid(self.rgate)
            ) * (w_x + w_h)
        if self.activation == 'tanh':
            h = self.tanh(w_w)
        else:
            h = self.relu(w_w)
        return h


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
