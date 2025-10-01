import torch
import torch._utils


class Padding4(torch.nn.Module):

    def __init__(self, input_channel):
        super(Padding4, self).__init__()
        self.requires_grad = False
        self.conv = torch.nn.ConvTranspose2d(input_channel, input_channel, 
            1, stride=2, padding=0, groups=input_channel, bias=False)
        torch.nn.init.constant_(self.conv.weight, 1)

    def forward(self, x):
        x1 = self.conv(x)
        y = torch.nn.functional.pad(x1, (1, 0, 1, 0))
        return y


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_channel': 4}]
