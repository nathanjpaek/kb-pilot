import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.utils.data
import torch.optim


class BatchDHCN(nn.Module):
    """docstring for BatchDHCN"""

    def __init__(self, embed_size=512, output_size=512, num_channel=2,
        conv_size=3, batch_norm=True):
        super(BatchDHCN, self).__init__()
        self.batch_norm = batch_norm
        self.embed_size = embed_size
        self.output_size = output_size
        self.num_channel = num_channel
        self.padding = nn.ZeroPad2d((0, conv_size - 1, conv_size - 1, 0))
        self.conv_1 = nn.Conv2d(self.num_channel, self.output_size, (
            conv_size, conv_size))
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x_conv_1 = self.conv_1(self.padding(x))
        x_conv_1 = F.relu(x_conv_1)
        return x_conv_1


def get_inputs():
    return [torch.rand([4, 2, 4, 4])]


def get_init_inputs():
    return [[], {}]
