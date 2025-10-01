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


def get_inputs():
    return [torch.rand([4, 4, 16, 16]), torch.rand([4, 4, 8, 8])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'planes': 4}]
