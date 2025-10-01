import torch
import torch.nn as nn
import torch.nn.init


class _TransitionUp(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(num_features, num_features,
            kernel_size=3, stride=2, padding=1)

    def forward(self, x, skip):
        self.deconv.padding = ((x.size(2) - 1) * self.deconv.stride[0] -
            skip.size(2) + self.deconv.kernel_size[0] + 1) // 2, ((x.size(3
            ) - 1) * self.deconv.stride[1] - skip.size(3) + self.deconv.
            kernel_size[1] + 1) // 2
        up = self.deconv(x, output_size=skip.size())
        return torch.cat([up, skip], 1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_features': 4}]
