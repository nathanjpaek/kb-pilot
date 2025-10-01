import torch
import torch.nn as nn
import torch.autograd


def pytorch_activation(name='relu'):
    if name == 'tanh':
        return nn.Tanh()
    if name == 'identity':
        return nn.Identity()
    if name == 'hardtanh':
        return nn.Hardtanh()
    if name == 'prelu':
        return nn.PReLU()
    if name == 'sigmoid':
        return nn.Sigmoid()
    if name == 'log_sigmoid':
        return nn.LogSigmoid()
    return nn.ReLU()


class ConvEncoder(nn.Module):

    def __init__(self, insz, outsz, filtsz, pdrop, activation_type='relu'):
        super(ConvEncoder, self).__init__()
        self.outsz = outsz
        pad = filtsz // 2
        self.conv = nn.Conv1d(insz, outsz, filtsz, padding=pad)
        self.act = pytorch_activation(activation_type)
        self.dropout = nn.Dropout(pdrop)

    def forward(self, input_bct):
        conv_out = self.act(self.conv(input_bct))
        return self.dropout(conv_out)


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'insz': 4, 'outsz': 4, 'filtsz': 4, 'pdrop': 0.5}]
