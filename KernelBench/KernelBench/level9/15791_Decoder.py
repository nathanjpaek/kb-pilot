import torch
import torch.nn
import torch.nn.functional as F
import torch.utils.data.dataset


class ResBlock(torch.nn.Module):

    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim is None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = torch.nn.Conv2d(indim, outdim, kernel_size=3,
                padding=1, stride=stride)
        self.conv1 = torch.nn.Conv2d(indim, outdim, kernel_size=3, padding=
            1, stride=stride)
        self.conv2 = torch.nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))
        if self.downsample is not None:
            x = self.downsample(x)
        return x + r


class Refine(torch.nn.Module):

    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = torch.nn.Conv2d(inplanes, planes, kernel_size=3,
            padding=1, stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, scale_factor=self.scale_factor, mode=
            'bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(torch.nn.Module):

    def __init__(self, mdim):
        super(Decoder, self).__init__()
        self.convFM = torch.nn.Conv2d(1024, mdim, kernel_size=3, padding=1,
            stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(512, mdim)
        self.RF2 = Refine(256, mdim)
        self.pred2 = torch.nn.Conv2d(mdim, 2, kernel_size=3, padding=1,
            stride=1)

    def forward(self, r4, r3, r2):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)
        p2 = self.pred2(F.relu(m2))
        p = F.interpolate(p2, scale_factor=4, mode='bilinear',
            align_corners=False)
        return p


def get_inputs():
    return [torch.rand([4, 1024, 64, 64]), torch.rand([4, 512, 128, 128]),
        torch.rand([4, 256, 256, 256])]


def get_init_inputs():
    return [[], {'mdim': 4}]
