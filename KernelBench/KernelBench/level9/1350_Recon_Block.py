import torch
import torch.nn as nn


class Recon_Block(nn.Module):

    def __init__(self, num_chans=64):
        super(Recon_Block, self).__init__()
        bias = True
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        self.conv13 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)
        self.relu14 = nn.PReLU()
        self.conv15 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)
        self.relu16 = nn.PReLU()
        self.conv17 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)

    def forward(self, x):
        res1 = x
        output = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        output = torch.add(output, res1)
        res2 = output
        output = self.relu8(self.conv7(self.relu6(self.conv5(output))))
        output = torch.add(output, res2)
        res3 = output
        output = self.relu12(self.conv11(self.relu10(self.conv9(output))))
        output = torch.add(output, res3)
        res4 = output
        output = self.relu16(self.conv15(self.relu14(self.conv13(output))))
        output = torch.add(output, res4)
        output = self.conv17(output)
        output = torch.add(output, res1)
        return output


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
