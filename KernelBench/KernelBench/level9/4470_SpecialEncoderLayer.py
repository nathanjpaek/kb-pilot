import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecialEncoderLayer(nn.Module):

    def __init__(self, heads, d_in, d_out, d_ff, p_drop=0.1):
        super(SpecialEncoderLayer, self).__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(d_in)
        self.proj_pair_1 = nn.Linear(d_in, heads // 2)
        self.proj_pair_2 = nn.Linear(d_in, heads // 2)
        self.proj_msa = nn.Linear(d_out, d_out)
        self.proj_out = nn.Linear(d_out, d_out)
        self.drop_1 = nn.Dropout(p_drop)
        self.drop_2 = nn.Dropout(p_drop)
        self.drop_3 = nn.Dropout(p_drop)
        self.linear1 = nn.Linear(d_out, d_ff)
        self.dropout = nn.Dropout(p_drop)
        self.linear2 = nn.Linear(d_ff, d_out)
        self.norm1 = nn.LayerNorm(d_out)
        self.norm2 = nn.LayerNorm(d_out)

    def forward(self, src, tgt):
        B, N, L = tgt.shape[:3]
        src = self.norm(src)
        attn_map_1 = F.softmax(self.proj_pair_1(src), dim=1).permute(0, 3, 1, 2
            )
        attn_map_2 = F.softmax(self.proj_pair_2(src), dim=2).permute(0, 3, 2, 1
            )
        attn_map = torch.cat((attn_map_1, attn_map_2), dim=1)
        attn_map = self.drop_1(attn_map).unsqueeze(1)
        tgt2 = self.norm1(tgt.view(B * N, L, -1)).view(B, N, L, -1)
        value = self.proj_msa(tgt2).permute(0, 3, 1, 2).contiguous().view(B,
            -1, self.heads, N, L)
        tgt2 = torch.matmul(value, attn_map).view(B, -1, N, L).permute(0, 2,
            3, 1)
        tgt2 = self.proj_out(tgt2)
        tgt = tgt + self.drop_2(tgt2)
        tgt2 = self.norm2(tgt.view(B * N, L, -1)).view(B, N, L, -1)
        tgt2 = self.linear2(self.dropout(F.relu_(self.linear1(tgt2))))
        tgt = tgt + self.drop_3(tgt2)
        return tgt


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'heads': 4, 'd_in': 4, 'd_out': 4, 'd_ff': 4}]
