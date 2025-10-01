import random
import torch
import torch.nn.init
from torch import nn
from torch.autograd import Variable
import torch.utils.data


class SCS_Cell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias,
        p_TD):
        super(SCS_Cell, self).__init__()
        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.p_TD = p_TD
        self.data_cnn = nn.Conv2d(in_channels=self.input_dim, out_channels=
            self.hidden_dim, kernel_size=self.kernel_size, padding=self.
            padding, bias=self.bias)
        self.ctrl_cnn = nn.Conv2d(in_channels=self.input_dim + self.
            hidden_dim, out_channels=self.hidden_dim, kernel_size=self.
            kernel_size, padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, cur_state):
        rate = random.random()
        c = cur_state
        data_x = input_tensor
        ctrl_x = input_tensor.detach() if rate < self.p_TD else input_tensor
        ctrl_in = torch.cat((c, ctrl_x), dim=1)
        data_out = torch.tanh(self.data_cnn(data_x))
        ctrl_out = torch.sigmoid(self.ctrl_cnn(ctrl_in))
        return ctrl_out * data_out, ctrl_out

    def init_hidden(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_dim, self.
            height, self.width))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': [4, 4], 'input_dim': 4, 'hidden_dim': 4,
        'kernel_size': [4, 4], 'bias': 4, 'p_TD': 4}]
