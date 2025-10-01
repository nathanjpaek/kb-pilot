import torch
import torch as th
import torch.utils.data


class BilinearUpsampler(th.nn.Module):

    def __init__(self, scale=2, channels=1):
        super(BilinearUpsampler, self).__init__()
        ksize = 2 * scale
        total_pad = ksize - scale // 2
        if scale % 2 == 1:
            ksize += 1
        self.pad = th.nn.ReplicationPad2d((1, 1, 1, 1))
        self.us_x = th.nn.ConvTranspose2d(channels, channels, (1, ksize),
            stride=(1, scale), padding=(0, total_pad), groups=channels,
            bias=False)
        self.us_y = th.nn.ConvTranspose2d(channels, channels, (ksize, 1),
            stride=(scale, 1), padding=(total_pad, 0), groups=channels,
            bias=False)
        k_idx = th.arange(0, ksize) + 0.5
        k_coord = k_idx / scale - ksize * 0.5 / scale
        k_weight = th.clamp(1.0 - th.abs(k_coord), min=0)
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
