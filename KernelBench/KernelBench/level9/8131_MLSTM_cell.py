import torch
import torch.nn as nn
from torch.autograd import Variable


class MLSTM_cell(nn.Module):

    def __init__(self, input_size, hidden_size, K, output_size):
        super(MLSTM_cell, self).__init__()
        self.hidden_size = hidden_size
        self.K = K
        self.output_size = output_size
        self.cgate = nn.Linear(input_size + hidden_size, hidden_size)
        self.igate = nn.Linear(input_size + hidden_size, hidden_size)
        self.fgate = nn.Linear(input_size + 2 * hidden_size, hidden_size)
        self.ogate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def get_ws(self, d):
        w = [1.0] * (self.K + 1)
        for i in range(0, self.K):
            w[self.K - i - 1] = w[self.K - i] * (i - d) / (i + 1)
        return torch.cat(w[0:self.K])

    def filter_d(self, celltensor, d):
        w = torch.ones(self.K, d.size(0), d.size(1), dtype=d.dtype, device=
            d.device)
        hidden_size = w.shape[2]
        batch_size = w.shape[1]
        for batch in range(batch_size):
            for hidden in range(hidden_size):
                w[:, batch, hidden] = self.get_ws(d[batch, hidden].view([1]))
        outputs = celltensor.mul(w).sum(dim=0)
        return outputs

    def forward(self, sample, hidden, celltensor, d_0):
        batch_size = sample.size(0)
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, dtype=sample
                .dtype, device=sample.device)
        if celltensor is None:
            celltensor = torch.zeros(self.K, batch_size, self.hidden_size,
                dtype=sample.dtype, device=sample.device)
        if d_0 is None:
            d_0 = torch.zeros(batch_size, self.hidden_size, dtype=sample.
                dtype, device=sample.device)
        combined = torch.cat((sample, hidden), 1)
        combined_d = torch.cat((sample, hidden, d_0), 1)
        d = self.fgate(combined_d)
        d = self.sigmoid(d) * 0.5
        first = -self.filter_d(celltensor, d)
        i_gate = self.igate(combined)
        o_gate = self.ogate(combined)
        i_gate = self.sigmoid(i_gate)
        o_gate = self.sigmoid(o_gate)
        c_tilde = self.cgate(combined)
        c_tilde = self.tanh(c_tilde)
        second = torch.mul(c_tilde, i_gate)
        cell = torch.add(first, second)
        hc = torch.cat([celltensor, cell.view([-1, cell.size(0), cell.size(
            1)])], 0)
        hc1 = hc[1:, :]
        hidden = torch.mul(self.tanh(cell), o_gate)
        output = self.output(hidden)
        return output, hidden, hc1, d

    def init_hidden(self):
        return Variable(torch.zeros(1, self.hidden_size))

    def init_cell(self):
        return Variable(torch.zeros(1, self.hidden_size))


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4, 4]),
        torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4, 'K': 4, 'output_size': 4}]
