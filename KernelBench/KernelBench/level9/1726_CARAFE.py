import torch
import torch.nn as nn
import torch.nn.functional as F


class CARAFE(nn.Module):

    def __init__(self, inC, outC, Kencoder=3, delta=2, Kup=5, Cm=64):
        super(CARAFE, self).__init__()
        self.Kencoder = Kencoder
        self.delta = delta
        self.Kup = Kup
        self.down = nn.Conv2d(in_channels=inC, out_channels=Cm, kernel_size=1)
        self.encoder = nn.Conv2d(64, self.delta ** 2 * self.Kup ** 2, self.
            Kencoder, 1, self.Kencoder // 2)
        self.out = nn.Conv2d(inC, outC, 1)

    def forward(self, in_tensor):
        N, C, H, W = in_tensor.size()
        kernel_tensor = self.down(in_tensor)
        kernel_tensor = self.encoder(kernel_tensor)
        kernel_tensor = F.pixel_shuffle(kernel_tensor, self.delta)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)
        kernel_tensor = kernel_tensor.unfold(2, self.delta, step=self.delta)
        kernel_tensor = kernel_tensor.unfold(3, self.delta, step=self.delta)
        kernel_tensor = kernel_tensor.reshape(N, self.Kup ** 2, H, W, self.
            delta ** 2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)
        in_tensor = F.pad(in_tensor, pad=(self.Kup // 2, self.Kup // 2, 
            self.Kup // 2, self.Kup // 2), mode='constant', value=0)
        in_tensor = in_tensor.unfold(dimension=2, size=self.Kup, step=1)
        in_tensor = in_tensor.unfold(3, self.Kup, step=1)
        in_tensor = in_tensor.reshape(N, C, H, W, -1)
        in_tensor = in_tensor.permute(0, 2, 3, 1, 4)
        out_tensor = torch.matmul(in_tensor, kernel_tensor)
        out_tensor = out_tensor.reshape(N, H, W, -1)
        out_tensor = out_tensor.permute(0, 3, 1, 2)
        out_tensor = F.pixel_shuffle(out_tensor, self.delta)
        out_tensor = self.out(out_tensor)
        return out_tensor


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'inC': 4, 'outC': 4}]
