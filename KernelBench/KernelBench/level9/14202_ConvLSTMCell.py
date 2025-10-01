import torch
from torch.autograd import Variable
import torch.nn as nn


class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_channels, kernel_size, bias=True
        ):
        super(ConvLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4
        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels, 4 *
            self.hidden_channels, self.kernel_size, 1, self.padding)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal(m.weight.data, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, input, h, c):
        combined = torch.cat((input, h), dim=1)
        A = self.conv(combined)
        ai, af, ao, ag = torch.split(A, A.size()[1] // self.num_features, dim=1
            )
        i = torch.sigmoid(ai)
        f = torch.sigmoid(af)
        o = torch.sigmoid(ao)
        g = torch.tanh(ag)
        new_c = f * c + i * g
        new_h = o * torch.tanh(new_c)
        return new_h, new_c, o

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        return Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1])
            ), Variable(torch.zeros(batch_size, hidden_c, shape[0], shape[1]))


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 3, 3])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4, 'kernel_size': 4}]
