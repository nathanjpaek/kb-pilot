import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def initialize(dims, connection_prob=1.0, shape=0.1, scale=1.0):
    w = np.random.gamma(shape, scale, size=dims)
    w *= np.random.rand(*dims) < connection_prob
    return np.float32(w)


class VanillaRNNCell(nn.Module):

    def __init__(self, input_size, hidden_size, nonlinearity='tanh', ct=False):
        super(VanillaRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.nonlinearity = nonlinearity
        self.ct = ct
        self.weight_ih = nn.Parameter(torch.zeros((input_size, hidden_size)))
        self.weight_hh = nn.Parameter(torch.zeros((hidden_size, hidden_size)))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.reset_parameters()
        if self.nonlinearity == 'tanh':
            self.act = F.tanh
        elif self.nonlinearity == 'relu':
            self.act = F.relu
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.
                nonlinearity))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, inp, hidden_in):
        if not self.ct:
            hidden_out = self.act(torch.matmul(inp, self.weight_ih) + torch
                .matmul(hidden_in, self.weight_hh) + self.bias)
        else:
            alpha = 0.1
            hidden_out = (1 - alpha) * hidden_in + alpha * self.act(torch.
                matmul(inp, self.weight_ih) + torch.matmul(hidden_in, self.
                weight_hh) + self.bias)
        return hidden_out

    def init_hidden(self, batch_s):
        return torch.zeros(batch_s, self.hidden_size)


class RNNSTSPConfig:

    def __init__(self, in_size, h_size):
        self.n_input = in_size
        self.n_hidden = h_size
        self.exc_inh_prop = 0.8
        self.balance_EI = True
        self.synapse_config = 'exc_dep_inh_fac'
        self.membrane_time_constant = 100
        self.tau_fast = 200
        self.tau_slow = 1500
        self.dt = 10
        self.dt_sec = self.dt / 1000
        self.trial_length = 2500
        self.num_time_steps = self.trial_length // self.dt
        if self.exc_inh_prop < 1:
            self.EI = True
        else:
            self.EI = False
        self.num_exc_units = int(np.round(self.n_hidden * self.exc_inh_prop))
        self.num_inh_units = self.n_hidden - self.num_exc_units
        self.EI_list = np.ones(self.n_hidden, dtype=np.float32)
        self.EI_list[-self.num_inh_units:] = -1.0
        self.ind_inh = np.where(self.EI_list == -1)[0]
        self.EI_matrix = np.diag(self.EI_list)
        self.alpha_neuron = np.float32(self.dt) / self.membrane_time_constant
        self.h0 = 0.1 * np.ones((1, self.n_hidden), dtype=np.float32)
        self.w_in0 = initialize([self.n_input, self.n_hidden], shape=0.2,
            scale=1.0)
        if self.EI:
            self.w_rnn0 = initialize([self.n_hidden, self.n_hidden])
            if self.balance_EI:
                self.w_rnn0[:, self.ind_inh] = initialize([self.n_hidden,
                    self.num_inh_units], shape=0.2, scale=1.0)
                self.w_rnn0[self.ind_inh, :] = initialize([self.
                    num_inh_units, self.n_hidden], shape=0.2, scale=1.0)
        else:
            self.w_rnn0 = 0.54 * np.eye(self.n_hidden)
        self.b_rnn0 = np.zeros(self.n_hidden, dtype=np.float32)
        self.w_rnn_mask = np.ones_like(self.w_rnn0)
        if self.EI:
            self.w_rnn_mask = np.ones((self.n_hidden, self.n_hidden), dtype
                =np.float32) - np.eye(self.n_hidden, dtype=np.float32)
        self.w_rnn0 *= self.w_rnn_mask
        synaptic_configurations = {'full': [('facilitating' if i % 2 == 0 else
            'depressing') for i in range(self.n_hidden)], 'fac': [
            'facilitating' for i in range(self.n_hidden)], 'dep': [
            'depressing' for i in range(self.n_hidden)], 'exc_fac': [(
            'facilitating' if self.EI_list[i] == 1 else 'static') for i in
            range(self.n_hidden)], 'exc_dep': [('depressing' if self.
            EI_list[i] == 1 else 'static') for i in range(self.n_hidden)],
            'inh_fac': [('facilitating' if self.EI_list[i] == -1 else
            'static') for i in range(self.n_hidden)], 'inh_dep': [(
            'depressing' if self.EI_list[i] == -1 else 'static') for i in
            range(self.n_hidden)], 'exc_dep_inh_fac': [('depressing' if 
            self.EI_list[i] == 1 else 'facilitating') for i in range(self.
            n_hidden)]}
        self.alpha_stf = np.ones((1, self.n_hidden), dtype=np.float32)
        self.alpha_std = np.ones((1, self.n_hidden), dtype=np.float32)
        self.U = np.ones((1, self.n_hidden), dtype=np.float32)
        self.syn_x_init = np.ones((1, self.n_hidden), dtype=np.float32)
        self.syn_u_init = 0.3 * np.ones((1, self.n_hidden), dtype=np.float32)
        self.dynamic_synapse = np.ones((1, self.n_hidden), dtype=np.float32)
        for i in range(self.n_hidden):
            if self.synapse_config not in synaptic_configurations.keys():
                self.dynamic_synapse[0, i] = 0
            elif synaptic_configurations[self.synapse_config][i
                ] == 'facilitating':
                self.alpha_stf[0, i] = self.dt / self.tau_slow
                self.alpha_std[0, i] = self.dt / self.tau_fast
                self.U[0, i] = 0.15
                self.syn_u_init[0, i] = self.U[0, i]
                self.dynamic_synapse[0, i] = 1
            elif synaptic_configurations[self.synapse_config][i
                ] == 'depressing':
                self.alpha_stf[0, i] = self.dt / self.tau_fast
                self.alpha_std[0, i] = self.dt / self.tau_slow
                self.U[0, i] = 0.45
                self.syn_u_init[0, i] = self.U[0, i]
                self.dynamic_synapse[0, i] = 1


class RNNSTSPCell(nn.Module):

    def __init__(self, input_size, hidden_size, nonlinearity='relu'):
        super(RNNSTSPCell, self).__init__()
        self.syncfg = RNNSTSPConfig(input_size, hidden_size)
        self.input_size = self.syncfg.n_input
        self.hidden_size = self.syncfg.n_hidden
        self.nonlinearity = nonlinearity
        self.weight_ih = nn.Parameter(torch.from_numpy(self.syncfg.w_in0))
        self.weight_hh = nn.Parameter(torch.from_numpy(self.syncfg.w_rnn0))
        self.bias = nn.Parameter(torch.from_numpy(self.syncfg.b_rnn0))
        self.EI = self.syncfg.EI
        if self.EI:
            self.EI_matrix = torch.from_numpy(self.syncfg.EI_matrix)
        self.w_rnn_mask = torch.from_numpy(self.syncfg.w_rnn_mask)
        self.alpha_stf = torch.from_numpy(self.syncfg.alpha_stf)
        self.alpha_std = torch.from_numpy(self.syncfg.alpha_std)
        self.U = torch.from_numpy(self.syncfg.U)
        self.dynamic_synapse = torch.from_numpy(self.syncfg.dynamic_synapse)
        if self.nonlinearity == 'tanh':
            self.act = F.tanh
        elif self.nonlinearity == 'relu':
            self.act = F.relu
        else:
            raise RuntimeError('Unknown nonlinearity: {}'.format(self.
                nonlinearity))

    def forward(self, inp, hidden_in):
        h_in, syn_x, syn_u = hidden_in
        syn_x = syn_x + (self.alpha_std * (1.0 - syn_x) - self.syncfg.
            dt_sec * syn_u * syn_x * h_in) * self.dynamic_synapse
        syn_u = syn_u + (self.alpha_stf * (self.U - syn_u) + self.syncfg.
            dt_sec * self.U * (1.0 - syn_u) * h_in) * self.dynamic_synapse
        syn_x = torch.clamp(syn_x, min=0.0, max=1.0)
        syn_u = torch.clamp(syn_u, min=0.0, max=1.0)
        h_post = syn_u * syn_x * h_in
        if self.EI:
            eff_rnn_w = self.w_rnn_mask * torch.matmul(self.EI_matrix, F.
                relu(self.weight_hh))
        else:
            eff_rnn_w = self.w_rnn_mask * self.weight_hh
        h_out = h_in * (1 - self.syncfg.alpha_neuron
            ) + self.syncfg.alpha_neuron * self.act(torch.matmul(inp, F.
            relu(self.weight_ih)) + torch.matmul(h_post, eff_rnn_w) + self.bias
            )
        hidden_out = h_out, syn_x, syn_u
        return hidden_out

    def init_hidden(self, batch_s):
        h_out_init = torch.from_numpy(self.syncfg.h0).repeat(batch_s, 1)
        syn_x_init = torch.from_numpy(self.syncfg.syn_x_init).repeat(batch_s, 1
            )
        syn_u_init = torch.from_numpy(self.syncfg.syn_u_init).repeat(batch_s, 1
            )
        hidden_init = h_out_init, syn_x_init, syn_u_init
        return hidden_init


class RNNModel(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, rnn_type):
        super(RNNModel, self).__init__()
        assert rnn_type in ['plainRNN', 'VanillaRNN', 'CTRNN', 'LSTM',
            'GRU', 'RNNSTSP'], 'Given RNN type must be implemented'
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type
        if self.rnn_type == 'plainRNN':
            self.rnn = nn.RNNCell(input_size, hidden_size)
        elif self.rnn_type == 'VanillaRNN':
            self.rnn = VanillaRNNCell(input_size, hidden_size)
        elif self.rnn_type == 'CTRNN':
            self.rnn = VanillaRNNCell(input_size, hidden_size, ct=True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTMCell(input_size, hidden_size)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRUCell(input_size, hidden_size)
        elif self.rnn_type == 'RNNSTSP':
            self.rnn = RNNSTSPCell(input_size, hidden_size)
        else:
            raise NotImplementedError('RNN cell not implemented')
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, inp, hidden_in):
        hid_out = self.rnn(inp, hidden_in)
        if self.rnn_type in ['LSTM', 'RNNSTSP']:
            rnn_out = hid_out[0]
        else:
            rnn_out = hid_out
        otp = self.out_layer(rnn_out)
        return otp, hid_out

    def init_hidden(self, batch_s):
        if self.rnn_type == 'LSTM':
            init_hid = torch.zeros(batch_s, self.hidden_size), torch.zeros(
                batch_s, self.hidden_size)
        elif self.rnn_type in ['plainRNN', 'GRU']:
            init_hid = torch.zeros(batch_s, self.hidden_size)
        elif self.rnn_type in ['VanillaRNN', 'CTRNN', 'RNNSTSP']:
            init_hid = self.rnn.init_hidden(batch_s)
        else:
            raise NotImplementedError('RNN init not implemented')
        return init_hid


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'output_size': 4,
        'rnn_type': 'plainRNN'}]
