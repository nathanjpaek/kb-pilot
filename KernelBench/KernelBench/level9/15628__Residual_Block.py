import torch
from torch import nn


class _Residual_Block(nn.Module):

    def __init__(self, num_chans=64):
        super(_Residual_Block, self).__init__()
        bias = True
        self.conv1 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride=
            1, padding=1, bias=bias)
        self.relu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(num_chans, num_chans * 2, kernel_size=3,
            stride=2, padding=1, bias=bias)
        self.relu6 = nn.PReLU()
        self.conv7 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3,
            stride=1, padding=1, bias=bias)
        self.relu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=3,
            stride=2, padding=1, bias=bias)
        self.relu10 = nn.PReLU()
        self.conv11 = nn.Conv2d(num_chans * 4, num_chans * 4, kernel_size=3,
            stride=1, padding=1, bias=bias)
        self.relu12 = nn.PReLU()
        self.conv13 = nn.Conv2d(num_chans * 4, num_chans * 8, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.up14 = nn.PixelShuffle(2)
        self.conv15 = nn.Conv2d(num_chans * 4, num_chans * 2, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.conv16 = nn.Conv2d(num_chans * 2, num_chans * 2, kernel_size=3,
            stride=1, padding=1, bias=bias)
        self.relu17 = nn.PReLU()
        self.conv18 = nn.Conv2d(num_chans * 2, num_chans * 4, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.up19 = nn.PixelShuffle(2)
        self.conv20 = nn.Conv2d(num_chans * 2, num_chans, kernel_size=1,
            stride=1, padding=0, bias=bias)
        self.conv21 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)
        self.relu22 = nn.PReLU()
        self.conv23 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)
        self.relu24 = nn.PReLU()
        self.conv25 = nn.Conv2d(num_chans, num_chans, kernel_size=3, stride
            =1, padding=1, bias=bias)

    def forward(self, x):
        res1 = x
        out = self.relu4(self.conv3(self.relu2(self.conv1(x))))
        out = torch.add(res1, out)
        cat1 = out
        out = self.relu6(self.conv5(out))
        res2 = out
        out = self.relu8(self.conv7(out))
        out = torch.add(res2, out)
        cat2 = out
        out = self.relu10(self.conv9(out))
        res3 = out
        out = self.relu12(self.conv11(out))
        out = torch.add(res3, out)
        out = self.up14(self.conv13(out))
        out = torch.cat([out, cat2], 1)
        out = self.conv15(out)
        res4 = out
        out = self.relu17(self.conv16(out))
        out = torch.add(res4, out)
        out = self.up19(self.conv18(out))
        out = torch.cat([out, cat1], 1)
        out = self.conv20(out)
        res5 = out
        out = self.relu24(self.conv23(self.relu22(self.conv21(out))))
        out = torch.add(res5, out)
        out = self.conv25(out)
        out = torch.add(out, res1)
        return out


def get_inputs():
    return [torch.rand([4, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
