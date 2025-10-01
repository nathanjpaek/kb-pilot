import torch
import torch.nn as nn


class Gate(nn.Module):

    def __init__(self, dhid, dfeature, init_range=0.1, init_dist='uniform',
        dropout=0.5):
        super(Gate, self).__init__()
        self.dhid = dhid
        self.dfeature = dfeature
        self.linear_z = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.linear_r = nn.Linear(self.dhid + self.dfeature, self.dfeature)
        self.linear_h_tilde = nn.Linear(self.dhid + self.dfeature, self.dhid)
        self.drop = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.init_weights(init_range, init_dist)

    def init_weights(self, init_range, init_dist):

        def init_w(data, init_dist):
            if init_dist == 'uniform':
                return data.uniform_(-init_range, init_range)
            elif init_dist == 'xavier':
                return nn.init.xavier_uniform(data)
        init_w(self.linear_z.weight.data, init_dist)
        init_w(self.linear_r.weight.data, init_dist)
        init_w(self.linear_h_tilde.weight.data, init_dist)
        self.linear_z.bias.data.fill_(0)
        self.linear_r.bias.data.fill_(0)
        self.linear_h_tilde.bias.data.fill_(0)

    def forward(self, h, features):
        z = self.sigmoid(self.linear_z(torch.cat((features, h), dim=1)))
        r = self.sigmoid(self.linear_r(torch.cat((features, h), dim=1)))
        h_tilde = self.tanh(self.linear_h_tilde(torch.cat((torch.mul(r,
            features), h), dim=1)))
        h_new = torch.mul(1 - z, h) + torch.mul(z, h_tilde)
        h_new = self.drop(h_new)
        return h_new


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'dhid': 4, 'dfeature': 4}]
