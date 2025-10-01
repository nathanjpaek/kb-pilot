import torch
import torch.utils.data
import torch.nn as nn


class SimpleLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.nf = input_dim
        self.hf = hidden_dim
        self.conv = nn.Conv2d(self.nf + self.hf, 4 * self.hf, 3, 1, 1, bias
            =True)

    def forward(self, input_tensor, h_cur, c_cur):
        if h_cur is None:
            tensor_size = input_tensor.size(2), input_tensor.size(3)
            h_cur = self._init_hidden(batch_size=input_tensor.size(0),
                tensor_size=tensor_size)
            c_cur = self._init_hidden(batch_size=input_tensor.size(0),
                tensor_size=tensor_size)
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hf, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def _init_hidden(self, batch_size, tensor_size):
        height, width = tensor_size
        return torch.zeros(batch_size, self.hf, height, width)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dim': 4}]
