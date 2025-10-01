import torch
import torch.nn as nn


class DisparityConv(nn.Module):

    def __init__(self, max_shift, output_nc):
        super().__init__()
        self.max_shift = int(max_shift)
        self.conv = nn.Conv2d(self.max_shift, output_nc, kernel_size=3,
            stride=1, padding=1, bias=True)

    def forward(self, x):
        b, c, h, w = x.shape
        Unfold = nn.Unfold(kernel_size=(h, w), stride=(1, 1))
        pad_clomns = x[:, :, :, 0:self.max_shift]
        x_cat = torch.cat([x, pad_clomns], dim=-1)
        patches = Unfold(x_cat)[:, :, 1:]
        patches = patches.permute([0, 2, 1])
        patches = torch.reshape(patches, [b, -1, c, h, w])
        x = x.unsqueeze(dim=1)
        diff = torch.abs(x - patches)
        diff = torch.mean(diff, dim=2, keepdim=False)
        out = self.conv(diff)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'max_shift': 4, 'output_nc': 4}]
