import torch


class CAMBlock(torch.nn.Module):

    def __init__(self, inplanes, redr, pool='full'):
        super(CAMBlock, self).__init__()
        self.planes = inplanes // redr
        self.poolingavg = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.poolingmax = torch.nn.AdaptiveMaxPool2d((1, 1))
        self.avglinear1 = torch.nn.Conv2d(inplanes, self.planes, (1, 1),
            padding=0)
        self.maxlinear1 = torch.nn.Conv2d(inplanes, self.planes, (1, 1),
            padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avglinear2 = torch.nn.Conv2d(self.planes, inplanes, (1, 1),
            padding=0)
        self.maxlinear2 = torch.nn.Conv2d(self.planes, inplanes, (1, 1),
            padding=0)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x):
        x1 = self.poolingavg(x)
        x2 = self.poolingmax(x)
        x1 = self.avglinear1(x1)
        x1 = self.relu(x1)
        x1 = self.avglinear2(x1)
        x2 = self.maxlinear1(x2)
        x2 = self.relu(x2)
        x2 = self.maxlinear2(x2)
        out = x1 + x2
        out = self.sigmod(out)
        out = x * out
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'redr': 4}]
