import torch
import torch.nn as nn


def match_score(s1, s2, mask1, mask2):
    """
    s1, s2:  batch_size * seq_len  * dim
    """
    _batch, seq_len, _dim = s1.shape
    s1 = s1 * mask1.eq(0).unsqueeze(2).float()
    s2 = s2 * mask2.eq(0).unsqueeze(2).float()
    s1 = s1.unsqueeze(2).repeat(1, 1, seq_len, 1)
    s2 = s2.unsqueeze(1).repeat(1, seq_len, 1, 1)
    a = s1 - s2
    a = torch.norm(a, dim=-1, p=2)
    return 1.0 / (1.0 + a)


class Wide_Conv(nn.Module):

    def __init__(self, seq_len, embeds_size, device='gpu'):
        super(Wide_Conv, self).__init__()
        self.seq_len = seq_len
        self.embeds_size = embeds_size
        self.W = nn.Parameter(torch.randn((seq_len, embeds_size)))
        nn.init.xavier_normal_(self.W)
        self.W
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3,
            padding=[1, 1], stride=1)
        self.tanh = nn.Tanh()

    def forward(self, sent1, sent2, mask1, mask2):
        """
        sent1, sent2: batch_size * seq_len * dim
        """
        A = match_score(sent1, sent2, mask1, mask2)
        attn_feature_map1 = A.matmul(self.W)
        attn_feature_map2 = A.transpose(1, 2).matmul(self.W)
        x1 = torch.cat([sent1.unsqueeze(1), attn_feature_map1.unsqueeze(1)], 1)
        x2 = torch.cat([sent2.unsqueeze(1), attn_feature_map2.unsqueeze(1)], 1)
        o1, o2 = self.conv(x1).squeeze(1), self.conv(x2).squeeze(1)
        o1, o2 = self.tanh(o1), self.tanh(o2)
        return o1, o2


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4]
        ), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'seq_len': 4, 'embeds_size': 4}]
