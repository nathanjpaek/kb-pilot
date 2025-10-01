import torch


class SEBlock(torch.nn.Module):

    def __init__(self, inplanes, redr, poolflag='avg'):
        super(SEBlock, self).__init__()
        if poolflag == 'max':
            self.pool = torch.nn.AdaptiveMaxPool2d((1, 1))
        if poolflag == 'avg':
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.replanes = inplanes // redr
        self.linear1 = torch.nn.Conv2d(inplanes, self.replanes, (1, 1),
            padding=0)
        self.relu = torch.nn.ReLU(inplace=True)
        self.linear2 = torch.nn.Conv2d(self.replanes, inplanes, (1, 1),
            padding=0)
        self.sigmod = torch.nn.Sigmoid()

    def forward(self, x):
        se = self.pool(x)
        se = self.linear1(se)
        se = self.relu(se)
        se = self.linear2(se)
        se = self.sigmod(se)
        x = se * x
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'redr': 4}]
