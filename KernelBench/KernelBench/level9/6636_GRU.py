import torch
import torch as tc
import torch.nn as nn


class Layer_Norm(nn.Module):

    def __init__(self, d_hid, eps=0.001):
        super(Layer_Norm, self).__init__()
        self.eps = eps
        self.g = nn.Parameter(tc.ones(d_hid), requires_grad=True)
        self.b = nn.Parameter(tc.zeros(d_hid), requires_grad=True)

    def forward(self, z):
        if z.size(-1) == 1:
            return z
        mu = tc.mean(z, dim=-1, keepdim=True)
        variance = tc.mean(tc.pow(z - mu, 2), dim=-1, keepdim=True)
        sigma = tc.sqrt(variance)
        z_norm = tc.div(z - mu, sigma + self.eps)
        if z_norm.is_cuda:
            z_norm = z_norm * self.g.expand_as(z_norm) + self.b.expand_as(
                z_norm)
        else:
            z_norm = z_norm * self.g.expand_as(z_norm) + self.b.expand_as(
                z_norm)
        return z_norm


class GRU(nn.Module):
    """
    Gated Recurrent Unit network with initial state as parameter:

        z_t = sigmoid((x_t dot W_xz + b_xz) + (h_{t-1} dot U_hz + b_hz))
        r_t = sigmoid((x_t dot W_xr + b_xr) + (h_{t-1} dot U_hr + b_xr))

        => zr_t = sigmoid((x_t dot W_xzr + b_xzr) + (h_{t-1} dot U_hzr + b_hzr))
        slice ...

        h_above = tanh(x_t dot W_xh + b_xh + (h_{t-1} dot U_hh + b_hh) * r_t)

        h_t = (1 - z_t) * h_above + z_t * h_{t-1}
        #h_t = (1 - z_t) * h_{t-1} + z_t * h_above

    all parameters are initialized in [-0.01, 0.01]
    """

    def __init__(self, input_size, hidden_size, enc_hid_size=None, with_ln=
        False, prefix='GRU', **kwargs):
        super(GRU, self).__init__()
        self.enc_hid_size = enc_hid_size
        self.hidden_size = hidden_size
        self.with_ln = with_ln
        self.prefix = prefix
        self.xh = nn.Linear(input_size, hidden_size)
        self.hh = nn.Linear(hidden_size, hidden_size)
        self.xrz = nn.Linear(input_size, 2 * hidden_size)
        self.hrz = nn.Linear(hidden_size, 2 * hidden_size)
        if self.enc_hid_size is not None:
            self.crz = nn.Linear(enc_hid_size, 2 * hidden_size)
            self.ch = nn.Linear(enc_hid_size, hidden_size)
        if self.with_ln is True:
            self.ln0 = Layer_Norm(2 * hidden_size)
            self.ln1 = Layer_Norm(2 * hidden_size)
            self.ln2 = Layer_Norm(hidden_size)
            self.ln3 = Layer_Norm(hidden_size)
            if self.enc_hid_size is not None:
                self.ln4 = Layer_Norm(2 * hidden_size)
                self.ln5 = Layer_Norm(hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
    """
        x_t: input at time t
        x_m: mask of x_t
        h_tm1: previous state
    """

    def forward(self, x_t, x_m, h_tm1, attend=None):
        x_rz_t, h_rz_tm1, x_h_t = self.xrz(x_t), self.hrz(h_tm1), self.xh(x_t)
        if self.with_ln is not True:
            if self.enc_hid_size is None:
                rz_t = x_rz_t + h_rz_tm1
            else:
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t
        else:
            x_rz_t, h_rz_tm1, x_h_t = self.ln0(x_rz_t), self.ln1(h_rz_tm1
                ), self.ln2(x_h_t)
            if self.enc_hid_size is None:
                rz_t = x_rz_t + h_rz_tm1
            else:
                a_rz_t, a_h_t = self.crz(attend), self.ch(attend)
                a_rz_t, a_h_t = self.ln4(a_rz_t), self.ln5(a_h_t)
                rz_t = x_rz_t + h_rz_tm1 + a_rz_t
        assert rz_t.dim() == 2
        rz_t = self.sigmoid(rz_t)
        r_t, z_t = rz_t[:, :self.hidden_size], rz_t[:, self.hidden_size:]
        h_h_tm1 = self.hh(r_t * h_tm1)
        if self.with_ln:
            h_h_tm1 = self.ln3(h_h_tm1)
        if self.enc_hid_size is None:
            h_h_tm1 = x_h_t + h_h_tm1
        else:
            h_h_tm1 = x_h_t + h_h_tm1 + a_h_t
        h_t_above = self.tanh(h_h_tm1)
        h_t = (1.0 - z_t) * h_tm1 + z_t * h_t_above
        if x_m is not None:
            h_t = x_m[:, None] * h_t + (1.0 - x_m[:, None]) * h_tm1
        return h_t


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'hidden_size': 4}]
