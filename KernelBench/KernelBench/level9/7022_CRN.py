import torch
import torch.nn.functional as F


class CRN(torch.nn.Module):

    def __init__(self, dim):
        super(CRN, self).__init__()
        self.h_w = 13, 13
        self.downsample = torch.nn.AdaptiveAvgPool2d(self.h_w)
        n_filters = [32, 32, 20]
        self.conv1 = torch.nn.Conv2d(dim, n_filters[0], 3, padding=1)
        self.conv2 = torch.nn.Conv2d(dim, n_filters[1], 5, padding=2)
        self.conv3 = torch.nn.Conv2d(dim, n_filters[2], 7, padding=3)
        self.conv_accum = torch.nn.Conv2d(sum(n_filters), 1, 1)
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',
                    nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        input_h_w = x.shape[2:]
        x = self.downsample(x)
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x))
        x3 = F.relu(self.conv3(x))
        x = torch.cat((x1, x2, x3), dim=1)
        x = F.relu(self.conv_accum(x))
        x = F.interpolate(x, input_h_w)
        assert x.shape[2:] == input_h_w
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
