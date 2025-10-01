import math
import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init


def Linear(in_features, out_features, dropout=0.0):
    m = nn.Linear(in_features, out_features)
    m.weight.data.normal_(mean=0, std=math.sqrt((1 - dropout) / in_features))
    m.bias.data.zero_()
    return nn.utils.weight_norm(m)


class AttentionLayer(nn.Module):

    def __init__(self, conv_channels, embed_dim):
        super(AttentionLayer, self).__init__()
        self.in_projection = Linear(conv_channels, embed_dim)
        self.out_projection = Linear(embed_dim, conv_channels)
        self.bmm = torch.bmm

    def forward(self, x, wordemb, imgsfeats):
        residual = x
        x = (self.in_projection(x) + wordemb) * math.sqrt(0.5)
        b, c, n = imgsfeats.size()
        y = imgsfeats.view(b, c, n)
        x = self.bmm(x, y)
        sz = x.size()
        x = F.softmax(x.view(sz[0] * sz[1], sz[2]))
        x = x.view(sz)
        attn_scores = x
        y = y.permute(0, 2, 1)
        x = self.bmm(x, y)
        s = y.size(1)
        x = x * (s * math.sqrt(1.0 / s))
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'conv_channels': 4, 'embed_dim': 4}]
