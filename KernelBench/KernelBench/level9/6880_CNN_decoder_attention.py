import torch
import torch.nn as nn
import torch.nn.init as init


class CNN_decoder_attention(nn.Module):

    def __init__(self, input_size, output_size, stride=2):
        super(CNN_decoder_attention, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(in_channels=int(self.input_size),
            out_channels=int(self.input_size), kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(in_channels=int(self.
            input_size), out_channels=int(self.output_size), kernel_size=3,
            stride=1, padding=1)
        self.deconv_attention = nn.ConvTranspose1d(in_channels=int(self.
            input_size), out_channels=int(self.input_size), kernel_size=1,
            stride=1, padding=0)
        self.bn_attention = nn.BatchNorm1d(int(self.input_size))
        self.relu_leaky = nn.LeakyReLU(0.2)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.
                    init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """

        :param
        x: batch * channel * length
        :return:
        """
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop1 = self.deconv_out(x)
        x_hop1_attention = self.deconv_attention(x)
        x_hop1_attention = self.relu(x_hop1_attention)
        x_hop1_attention = torch.matmul(x_hop1_attention, x_hop1_attention.
            view(-1, x_hop1_attention.size(2), x_hop1_attention.size(1)))
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop2 = self.deconv_out(x)
        x_hop2_attention = self.deconv_attention(x)
        x_hop2_attention = self.relu(x_hop2_attention)
        x_hop2_attention = torch.matmul(x_hop2_attention, x_hop2_attention.
            view(-1, x_hop2_attention.size(2), x_hop2_attention.size(1)))
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop3 = self.deconv_out(x)
        x_hop3_attention = self.deconv_attention(x)
        x_hop3_attention = self.relu(x_hop3_attention)
        x_hop3_attention = torch.matmul(x_hop3_attention, x_hop3_attention.
            view(-1, x_hop3_attention.size(2), x_hop3_attention.size(1)))
        return (x_hop1, x_hop2, x_hop3, x_hop1_attention, x_hop2_attention,
            x_hop3_attention)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_size': 4, 'output_size': 4}]
