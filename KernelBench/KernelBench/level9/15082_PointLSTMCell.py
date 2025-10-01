import torch
import torch.nn as nn


class PointLSTMCell(nn.Module):

    def __init__(self, pts_num, in_channels, hidden_dim, offset_dim, bias):
        super(PointLSTMCell, self).__init__()
        self.bias = bias
        self.pts_num = pts_num
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.offset_dim = offset_dim
        self.pool = nn.Sequential(nn.AdaptiveMaxPool2d((None, 1)))
        self.conv = nn.Conv2d(in_channels=self.in_channels + self.
            offset_dim + self.hidden_dim, out_channels=4 * self.hidden_dim,
            kernel_size=(1, 1), bias=self.bias)

    def forward(self, input_tensor, hidden_state, cell_state):
        hidden_state[:, :4] -= input_tensor[:, :4]
        combined = torch.cat([input_tensor, hidden_state], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim,
            dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * cell_state + i * g
        h_next = o * torch.tanh(c_next)
        return self.pool(h_next), self.pool(c_next)

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1
            ), torch.zeros(batch_size, self.hidden_dim, self.pts_num, 1)


def get_inputs():
    return [torch.rand([4, 8, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'pts_num': 4, 'in_channels': 4, 'hidden_dim': 4,
        'offset_dim': 4, 'bias': 4}]
