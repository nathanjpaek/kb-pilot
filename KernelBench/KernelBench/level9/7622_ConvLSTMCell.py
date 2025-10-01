import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Implementation of the Basic ConvLSTM.
    No peephole connection, no forget gate.

    ConvLSTM:
        x - input
        h - hidden representation
        c - memory cell
        f - forget gate
        o - output gate

    Reference:Convolutional LSTM Network: A Machine Learning Approach for Precipitation
    Nowcasting
    """

    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = int((kernel_size - 1) / 2)
        self.W_i = nn.Conv2d(self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.W_f = nn.Conv2d(self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.W_o = nn.Conv2d(self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.W_c = nn.Conv2d(self.input_channels, self.hidden_channels,
            self.kernel_size, 1, self.padding, bias=True)
        self.reset_parameters()

    def forward(self, inputs, c):
        i_t = torch.sigmoid(self.W_i(inputs))
        f_t = torch.sigmoid(self.W_f(inputs))
        o_t = torch.sigmoid(self.W_o(inputs))
        c_t = f_t * c + i_t * torch.tanh(self.W_c(inputs))
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

    def reset_parameters(self):
        self.W_i.reset_parameters()
        self.W_f.reset_parameters()
        self.W_o.reset_parameters()
        self.W_c.reset_parameters()


def get_inputs():
    return [torch.rand([4, 4, 3, 3]), torch.rand([4, 4, 2, 2])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4, 'kernel_size': 4}]
