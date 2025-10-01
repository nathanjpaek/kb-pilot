import torch
import torch.utils.data
import torch.utils
import torch.utils.checkpoint


class Conv(torch.nn.Module):

    def __init__(self, in_dim, out_dim, filter_length, stride):
        super(Conv, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=in_dim, out_channels=
            out_dim, kernel_size=filter_length, stride=stride)
        self.filter_length = filter_length

    def forward(self, x):
        out = x.transpose(1, 2)
        left_padding = int(self.filter_length / 2)
        right_padding = int(self.filter_length / 2)
        out = torch.nn.functional.pad(out, (left_padding, right_padding))
        out = self.conv(out)
        out = out.transpose(1, 2)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'filter_length': 4, 'stride': 1}]
