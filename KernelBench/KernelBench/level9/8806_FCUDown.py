import torch
from functools import partial
from torch import nn


class FCUDown(nn.Module):
    """ CNN feature maps -> Transformer patch embeddings
    """

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU,
        norm_layer=partial(nn.LayerNorm, eps=1e-06)):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1,
            stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=
            dw_stride)
        self.ln = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x)
        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)
        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'inplanes': 4, 'outplanes': 4, 'dw_stride': 1}]
