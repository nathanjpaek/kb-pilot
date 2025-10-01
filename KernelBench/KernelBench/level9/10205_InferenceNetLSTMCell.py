import torch
import torch.nn as nn


class InferenceNetLSTMCell(nn.Module):

    def __init__(self, z_dim: 'int', input_dim: 'int', hidden_hat_dim:
        'int', hidden_dim: 'int'):
        super(InferenceNetLSTMCell, self).__init__()
        self.w_hh = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hx = nn.Linear(hidden_hat_dim, z_dim)
        self.w_hb = nn.Linear(hidden_hat_dim, z_dim)
        self.W_hz = nn.Linear(z_dim, 4 * hidden_dim, bias=False)
        self.W_xz = nn.Linear(z_dim, 4 * hidden_dim, bias=False)
        self.b = nn.Linear(z_dim, 4 * hidden_dim)
        self.Wh = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.Wx = nn.Linear(input_dim, 4 * hidden_dim)
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.norm_h = nn.LayerNorm(hidden_dim)
        self.norm_c = nn.LayerNorm(hidden_dim)

    def forward(self, h_t, c, h_t_hat, inf_inputs):
        z_h = self.w_hh(h_t_hat)
        z_x = self.w_hx(h_t_hat)
        z_bias = self.w_hb(h_t_hat)
        d_z_h = self.W_hz(z_h)
        d_z_x = self.W_xz(z_x)
        b_z_b = self.b(z_bias)
        ifgo = d_z_h * self.Wh(h_t) + d_z_x * self.Wx(inf_inputs) + b_z_b
        i, f, g, o = torch.chunk(ifgo, 4, -1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.sigmoid(g)
        o = torch.sigmoid(o)
        new_c = f * c + i * c
        new_h = o * torch.tanh(new_c)
        new_h = self.dropout(new_h)
        new_h = self.norm_h(new_h)
        new_c = self.norm_c(new_c)
        return new_h, new_c


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'z_dim': 4, 'input_dim': 4, 'hidden_hat_dim': 4,
        'hidden_dim': 4}]
