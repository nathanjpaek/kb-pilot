import torch
import torch.nn as nn


class AttentionGateBlock(nn.Module):

    def __init__(self, chns_l, chns_h):
        """
        chns_l: channel number of low-level features from the encoder
        chns_h: channel number of high-level features from the decoder
        """
        super(AttentionGateBlock, self).__init__()
        self.in_chns_l = chns_l
        self.in_chns_h = chns_h
        self.out_chns = int(min(self.in_chns_l, self.in_chns_h) / 2)
        self.conv1_l = nn.Conv2d(self.in_chns_l, self.out_chns, kernel_size
            =1, bias=True)
        self.conv1_h = nn.Conv2d(self.in_chns_h, self.out_chns, kernel_size
            =1, bias=True)
        self.conv2 = nn.Conv2d(self.out_chns, 1, kernel_size=1, bias=True)
        self.act1 = nn.ReLU()
        self.act2 = nn.Sigmoid()

    def forward(self, x_l, x_h):
        input_shape = list(x_l.shape)
        gate_shape = list(x_h.shape)
        x_l_reshape = nn.functional.interpolate(x_l, size=gate_shape[2:],
            mode='bilinear')
        f_l = self.conv1_l(x_l_reshape)
        f_h = self.conv1_h(x_h)
        f = f_l + f_h
        f = self.act1(f)
        f = self.conv2(f)
        att = self.act2(f)
        att = nn.functional.interpolate(att, size=input_shape[2:], mode=
            'bilinear')
        output = att * x_l
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'chns_l': 4, 'chns_h': 4}]
