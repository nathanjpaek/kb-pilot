import torch
import torch.nn as nn
import torch._C
import torch.serialization


class ChannelPool(nn.Module):

    def forward(self, x):
        channel_out = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.
            mean(x, 1).unsqueeze(1)), dim=1)
        return channel_out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
