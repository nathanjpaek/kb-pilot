import logging
import torch
import torch.nn as nn
from torch.autograd import Variable


class BottleneckLSTMCell(nn.Module):
    """ Creates a LSTM layer cell
    Arguments:
        input_channels : variable used to contain value of number of channels in input
        hidden_channels : variable used to contain value of number of channels in the hidden state of LSTM cell
    """

    def __init__(self, input_channels, hidden_channels, device):
        super(BottleneckLSTMCell, self).__init__()
        assert hidden_channels % 2 == 0
        self.device = device
        self.input_channels = int(input_channels)
        self.hidden_channels = int(hidden_channels)
        self.num_features = 4
        self.W = nn.Conv2d(in_channels=self.input_channels, out_channels=
            self.input_channels, kernel_size=3, groups=self.input_channels,
            stride=1, padding=1)
        self.Wy = nn.Conv2d(int(self.input_channels + self.hidden_channels),
            self.hidden_channels, kernel_size=1)
        self.Wi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 
            1, 1, groups=self.hidden_channels, bias=False)
        self.Wbi = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1,
            1, 0, bias=False)
        self.Wbf = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1,
            1, 0, bias=False)
        self.Wbc = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1,
            1, 0, bias=False)
        self.Wbo = nn.Conv2d(self.hidden_channels, self.hidden_channels, 1,
            1, 0, bias=False)
        self.relu = nn.ReLU6()
        logging.info('Initializing weights of lstm')
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Returns:
            initialized weights of the model
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, h, c):
        """
        Arguments:
            x : input tensor
            h : hidden state tensor
            c : cell state tensor
        Returns:
            output tensor after LSTM cell
        """
        x = self.W(x)
        y = torch.cat((x, h), 1)
        i = self.Wy(y)
        b = self.Wi(i)
        ci = torch.sigmoid(self.Wbi(b))
        cf = torch.sigmoid(self.Wbf(b))
        cc = cf * c + ci * self.relu(self.Wbc(b))
        co = torch.sigmoid(self.Wbo(b))
        ch = co * self.relu(cc)
        return ch, cc

    def init_hidden(self, batch_size, hidden, shape):
        """
        Arguments:
            batch_size : an int variable having value of batch size while training
            hidden : an int variable having value of number of channels in hidden state
            shape : an array containing shape of the hidden and cell state
            device : CPU or GPU id string device
        Returns:
            cell state and hidden state
        """
        if str(self.device) == 'cpu':
            return Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])
                ), Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])
                )
        else:
            return Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])
                ), Variable(torch.zeros(batch_size, hidden, shape[0], shape[1])
                )


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channels': 4, 'hidden_channels': 4, 'device': 0}]
