import torch
import torch.nn as nn


class SAM(nn.Module):

    def __init__(self, channels_in):
        super(SAM, self).__init__()
        self.channels_in = channels_in
        self.avg_pool = nn.AvgPool3d(kernel_size=(self.channels_in, 1, 1))
        self.max_pool = nn.MaxPool3d(kernel_size=(self.channels_in, 1, 1))
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7,
            stride=1, padding=3)

    def forward(self, x, save_attention=False):
        spat_att = torch.cat(tensors=(self.avg_pool(x), self.max_pool(x)),
            dim=1)
        spat_att = torch.sigmoid(self.conv1(spat_att))
        if save_attention:
            torch.save(spat_att,
                f'tmp/cbam-attention_spatial_{spat_att.shape[-2]}-{spat_att.shape[-1]}.pt'
                )
        return spat_att

    def initialize_weights(self):
        nn.init.normal_(self.conv1.weight.data, mean=0.0, std=0.02)
        nn.init.constant_(self.conv1.bias.data, 0.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'channels_in': 4}]
