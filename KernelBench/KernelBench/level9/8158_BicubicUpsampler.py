import torch
import torch as th
import torch.utils.data


class BicubicUpsampler(th.nn.Module):

    def __init__(self, scale=2, channels=1):
        super(BicubicUpsampler, self).__init__()
        ksize = 2 * scale * 2
        total_pad = ksize - scale // 2
        if scale % 2 == 1:
            ksize += 1
        self.pad = th.nn.ReplicationPad2d((2, 2, 2, 2))
        self.us_x = th.nn.ConvTranspose2d(channels, channels, (1, ksize),
            stride=(1, scale), padding=(0, total_pad), groups=channels,
            bias=False)
        self.us_y = th.nn.ConvTranspose2d(channels, channels, (ksize, 1),
            stride=(scale, 1), padding=(total_pad, 0), groups=channels,
            bias=False)
        k_idx = th.arange(0, ksize) + 0.5
        k_coord = k_idx / scale - ksize * 0.5 / scale
        absx = th.abs(k_coord)
        absx2 = absx.pow(2)
        absx3 = absx.pow(3)
        k_weight = th.zeros(ksize)
        k_weight += (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2.0) * ((absx >
            1.0) & (absx < 2.0))
        k_weight += (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
        for c in range(channels):
            self.us_x.weight.data[c, 0, 0, :].copy_(k_weight)
            self.us_y.weight.data[c, 0, :, 0].copy_(k_weight)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.pad(x)
        x = self.us_x(x)
        x = self.us_y(x)
        return x


def get_inputs():
    return [torch.rand([4, 1, 4, 4])]


def get_init_inputs():
    return [[], {}]
