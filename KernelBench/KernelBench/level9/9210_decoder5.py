import torch
import torch.nn as nn


class decoder5(nn.Module):

    def __init__(self, d=None):
        super(decoder5, self).__init__()
        self.reflecPad15 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv15 = nn.Conv2d(512, 512, 3, 1, 0)
        if d:
            self.conv15.weight = torch.nn.Parameter(d.get(1).weight.float())
            self.conv15.bias = torch.nn.Parameter(d.get(1).bias.float())
        self.relu15 = nn.ReLU(inplace=True)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad16 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv16 = nn.Conv2d(512, 512, 3, 1, 0)
        if d:
            self.conv16.weight = torch.nn.Parameter(d.get(5).weight.float())
            self.conv16.bias = torch.nn.Parameter(d.get(5).bias.float())
        self.relu16 = nn.ReLU(inplace=True)
        self.reflecPad17 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv17 = nn.Conv2d(512, 512, 3, 1, 0)
        if d:
            self.conv17.weight = torch.nn.Parameter(d.get(8).weight.float())
            self.conv17.bias = torch.nn.Parameter(d.get(8).bias.float())
        self.relu17 = nn.ReLU(inplace=True)
        self.reflecPad18 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv18 = nn.Conv2d(512, 512, 3, 1, 0)
        if d:
            self.conv18.weight = torch.nn.Parameter(d.get(11).weight.float())
            self.conv18.bias = torch.nn.Parameter(d.get(11).bias.float())
        self.relu18 = nn.ReLU(inplace=True)
        self.reflecPad19 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv19 = nn.Conv2d(512, 256, 3, 1, 0)
        if d:
            self.conv19.weight = torch.nn.Parameter(d.get(14).weight.float())
            self.conv19.bias = torch.nn.Parameter(d.get(14).bias.float())
        self.relu19 = nn.ReLU(inplace=True)
        self.unpool2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad20 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv20 = nn.Conv2d(256, 256, 3, 1, 0)
        if d:
            self.conv20.weight = torch.nn.Parameter(d.get(18).weight.float())
            self.conv20.bias = torch.nn.Parameter(d.get(18).bias.float())
        self.relu20 = nn.ReLU(inplace=True)
        self.reflecPad21 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv21 = nn.Conv2d(256, 256, 3, 1, 0)
        if d:
            self.conv21.weight = torch.nn.Parameter(d.get(21).weight.float())
            self.conv21.bias = torch.nn.Parameter(d.get(21).bias.float())
        self.relu21 = nn.ReLU(inplace=True)
        self.reflecPad22 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv22 = nn.Conv2d(256, 256, 3, 1, 0)
        if d:
            self.conv22.weight = torch.nn.Parameter(d.get(24).weight.float())
            self.conv22.bias = torch.nn.Parameter(d.get(24).bias.float())
        self.relu22 = nn.ReLU(inplace=True)
        self.reflecPad23 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv23 = nn.Conv2d(256, 128, 3, 1, 0)
        if d:
            self.conv23.weight = torch.nn.Parameter(d.get(27).weight.float())
            self.conv23.bias = torch.nn.Parameter(d.get(27).bias.float())
        self.relu23 = nn.ReLU(inplace=True)
        self.unpool3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad24 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv24 = nn.Conv2d(128, 128, 3, 1, 0)
        if d:
            self.conv24.weight = torch.nn.Parameter(d.get(31).weight.float())
            self.conv24.bias = torch.nn.Parameter(d.get(31).bias.float())
        self.relu24 = nn.ReLU(inplace=True)
        self.reflecPad25 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv25 = nn.Conv2d(128, 64, 3, 1, 0)
        if d:
            self.conv25.weight = torch.nn.Parameter(d.get(34).weight.float())
            self.conv25.bias = torch.nn.Parameter(d.get(34).bias.float())
        self.relu25 = nn.ReLU(inplace=True)
        self.unpool4 = nn.UpsamplingNearest2d(scale_factor=2)
        self.reflecPad26 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv26 = nn.Conv2d(64, 64, 3, 1, 0)
        if d:
            self.conv26.weight = torch.nn.Parameter(d.get(38).weight.float())
            self.conv26.bias = torch.nn.Parameter(d.get(38).bias.float())
        self.relu26 = nn.ReLU(inplace=True)
        self.reflecPad27 = nn.ReflectionPad2d((1, 1, 1, 1))
        self.conv27 = nn.Conv2d(64, 3, 3, 1, 0)
        if d:
            self.conv27.weight = torch.nn.Parameter(d.get(41).weight.float())
            self.conv27.bias = torch.nn.Parameter(d.get(41).bias.float())

    def forward(self, x):
        out = self.reflecPad15(x)
        out = self.conv15(out)
        out = self.relu15(out)
        out = self.unpool(out)
        out = self.reflecPad16(out)
        out = self.conv16(out)
        out = self.relu16(out)
        out = self.reflecPad17(out)
        out = self.conv17(out)
        out = self.relu17(out)
        out = self.reflecPad18(out)
        out = self.conv18(out)
        out = self.relu18(out)
        out = self.reflecPad19(out)
        out = self.conv19(out)
        out = self.relu19(out)
        out = self.unpool2(out)
        out = self.reflecPad20(out)
        out = self.conv20(out)
        out = self.relu20(out)
        out = self.reflecPad21(out)
        out = self.conv21(out)
        out = self.relu21(out)
        out = self.reflecPad22(out)
        out = self.conv22(out)
        out = self.relu22(out)
        out = self.reflecPad23(out)
        out = self.conv23(out)
        out = self.relu23(out)
        out = self.unpool3(out)
        out = self.reflecPad24(out)
        out = self.conv24(out)
        out = self.relu24(out)
        out = self.reflecPad25(out)
        out = self.conv25(out)
        out = self.relu25(out)
        out = self.unpool4(out)
        out = self.reflecPad26(out)
        out = self.conv26(out)
        out = self.relu26(out)
        out = self.reflecPad27(out)
        out = self.conv27(out)
        return out


def get_inputs():
    return [torch.rand([4, 512, 4, 4])]


def get_init_inputs():
    return [[], {}]
