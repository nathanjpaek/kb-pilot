import torch
import torch.nn as nn
import torch.utils.data


def sample_K_laplace_MN(MAP, upper_Qinv, lower_HHinv):
    Z = MAP.data.new(MAP.size()).normal_(mean=0, std=1)
    all_mtx_sample = MAP + torch.matmul(torch.matmul(lower_HHinv, Z),
        upper_Qinv)
    weight_mtx_sample = all_mtx_sample[:, :-1]
    bias_mtx_sample = all_mtx_sample[:, -1]
    return weight_mtx_sample, bias_mtx_sample


class Linear_2L_KFRA(nn.Module):

    def __init__(self, input_dim, output_dim, n_hid):
        super(Linear_2L_KFRA, self).__init__()
        self.n_hid = n_hid
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, self.n_hid)
        self.fc2 = nn.Linear(self.n_hid, self.n_hid)
        self.fc3 = nn.Linear(self.n_hid, output_dim)
        self.act = nn.ReLU(inplace=True)
        self.one = None
        self.a2 = None
        self.h2 = None
        self.a1 = None
        self.h1 = None
        self.a0 = None

    def forward(self, x):
        self.one = x.new(x.shape[0], 1).fill_(1)
        a0 = x.view(-1, self.input_dim)
        self.a0 = torch.cat((a0.data, self.one), dim=1)
        h1 = self.fc1(a0)
        self.h1 = h1.data
        a1 = self.act(h1)
        self.a1 = torch.cat((a1.data, self.one), dim=1)
        h2 = self.fc2(a1)
        self.h2 = h2.data
        a2 = self.act(h2)
        self.a2 = torch.cat((a2.data, self.one), dim=1)
        h3 = self.fc3(a2)
        return h3

    def sample_predict(self, x, Nsamples, Qinv1, HHinv1, MAP1, Qinv2,
        HHinv2, MAP2, Qinv3, HHinv3, MAP3):
        predictions = x.data.new(Nsamples, x.shape[0], self.output_dim)
        x = x.view(-1, self.input_dim)
        for i in range(Nsamples):
            w1, b1 = sample_K_laplace_MN(MAP1, Qinv1, HHinv1)
            a = torch.matmul(x, torch.t(w1)) + b1.unsqueeze(0)
            a = self.act(a)
            w2, b2 = sample_K_laplace_MN(MAP2, Qinv2, HHinv2)
            a = torch.matmul(a, torch.t(w2)) + b2.unsqueeze(0)
            a = self.act(a)
            w3, b3 = sample_K_laplace_MN(MAP3, Qinv3, HHinv3)
            y = torch.matmul(a, torch.t(w3)) + b3.unsqueeze(0)
            predictions[i] = y
        return predictions


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'output_dim': 4, 'n_hid': 4}]
