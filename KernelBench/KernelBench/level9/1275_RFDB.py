import torch
import torch.nn as nn
import torch.nn.functional as F


class ESA(nn.Module):

    def __init__(self, num_feat=50, conv=nn.Conv2d, p=0.25):
        super(ESA, self).__init__()
        f = num_feat // 4
        BSConvS_kwargs = {}
        if conv.__name__ == 'BSConvS':
            BSConvS_kwargs = {'p': p}
        self.conv1 = nn.Linear(num_feat, f)
        self.conv_f = nn.Linear(f, f)
        self.maxPooling = nn.MaxPool2d(kernel_size=7, stride=3)
        self.conv_max = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv2 = nn.Conv2d(f, f, 3, 2, 0)
        self.conv3 = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv3_ = conv(f, f, kernel_size=3, **BSConvS_kwargs)
        self.conv4 = nn.Linear(f, num_feat)
        self.sigmoid = nn.Sigmoid()
        self.GELU = nn.GELU()

    def forward(self, input):
        x = input.permute(0, 2, 3, 1)
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_.permute(0, 3, 1, 2))
        v_max = self.maxPooling(c1)
        v_range = self.GELU(self.conv_max(v_max))
        c3 = self.GELU(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (input.size(2), input.size(3)), mode=
            'bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3.permute(0, 2, 3, 1) + cf)
        m = self.sigmoid(c4.permute(0, 3, 1, 2))
        return input * m


class RFDB(nn.Module):

    def __init__(self, in_channels, out_channels, distillation_rate=0.25,
        conv=nn.Conv2d, p=0.25):
        super(RFDB, self).__init__()
        kwargs = {'padding': 1}
        if conv.__name__ == 'BSConvS':
            kwargs = {'p': p}
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = nn.Linear(in_channels, self.rc)
        self.c1_r = conv(in_channels, self.rc, kernel_size=3, **kwargs)
        self.c2_d = nn.Linear(self.remaining_channels, self.rc)
        self.c2_r = conv(self.remaining_channels, self.rc, kernel_size=3,
            **kwargs)
        self.c3_d = nn.Linear(self.remaining_channels, self.rc)
        self.c3_r = conv(self.remaining_channels, self.rc, kernel_size=3,
            **kwargs)
        self.c4 = conv(self.remaining_channels, in_channels, kernel_size=3,
            **kwargs)
        self.act = nn.GELU()
        self.alpha1 = nn.Parameter(torch.ones(1, in_channels))
        self.alpha2 = nn.Parameter(torch.ones(1, in_channels))
        self.alpha3 = nn.Parameter(torch.ones(1, in_channels))
        self.alpha4 = nn.Parameter(torch.ones(1, in_channels))
        self.esa = ESA(in_channels, conv)
        self.conv_out = nn.Linear(in_channels, out_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input.permute(0, 2, 3, 1)))
        r_c1 = self.c1_r(input)
        r_c1 = self.act(r_c1 + input)
        distilled_c2 = self.act(self.c2_d(r_c1.permute(0, 2, 3, 1)))
        r_c2 = self.c2_r(r_c1)
        r_c2 = self.act(r_c2 + r_c1)
        distilled_c3 = self.act(self.c3_d(r_c2.permute(0, 2, 3, 1)))
        r_c3 = self.c3_r(r_c2)
        r_c3 = self.act(r_c3 + r_c2)
        r_c4 = self.act(self.c4(r_c3))
        out = (distilled_c1 * self.alpha1 + distilled_c2 * self.alpha2 + 
            distilled_c3 * self.alpha3 + r_c4.permute(0, 2, 3, 1) * self.alpha4
            )
        out_fused = self.esa(out.permute(0, 3, 1, 2))
        out_fused = self.conv_out(out_fused.permute(0, 2, 3, 1))
        return out_fused.permute(0, 3, 1, 2) + input


def get_inputs():
    return [torch.rand([4, 4, 64, 64])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
