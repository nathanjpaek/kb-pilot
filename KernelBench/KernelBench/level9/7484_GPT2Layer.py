from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_ipt: 'int', n_head: 'int', dropout_p: 'float'=0.1):
        super(MultiHeadSelfAttention, self).__init__()
        self.qkv_linear = nn.Linear(d_ipt, d_ipt * 3, True)
        self.n_head = n_head
        self.output_linear = nn.Linear(d_ipt, d_ipt, True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src: 'torch.FloatTensor', attn_mask: 'torch.FloatTensor'
        ) ->torch.FloatTensor:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
        if attn_mask.dim() == 3:
            attn_mask = attn_mask.unsqueeze(1)
        q, k, v = self.qkv_linear(src).chunk(3, dim=-1)
        q = q.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 1, 3)
        k = k.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 3, 1)
        v = v.contiguous().view(src.shape[0], src.shape[1], self.n_head, 
            src.shape[2] // self.n_head).permute(0, 2, 1, 3)
        attn_weights = torch.matmul(q, k)
        attn_weights = attn_weights * float(src.shape[2] // self.n_head
            ) ** -0.5
        attn_weights = attn_weights * attn_mask + (attn_mask - 1) * 10000.0
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous().view(src
            .shape)
        attn_output = self.output_linear(attn_output)
        return attn_output


class GPT2Layer(nn.Module):

    def __init__(self, config: 'GPT2Config'):
        super(GPT2Layer, self).__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.self_attn = MultiHeadSelfAttention(d_ipt=config.hidden_size,
            n_head=config.n_head, dropout_p=config.drop_out)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        self.intermediate_linear1 = nn.Linear(config.hidden_size, config.
            d_intermediate, True)
        self.intermediate_linear2 = nn.Linear(config.d_intermediate, config
            .hidden_size, True)
        self.dropout = nn.Dropout(config.drop_out)
        self.dropout1 = nn.Dropout(config.drop_out)
        self.dropout2 = nn.Dropout(config.drop_out)

    def forward(self, src: 'torch.FloatTensor', src_mask: 'torch.FloatTensor'
        ) ->torch.FloatTensor:
        src1 = self.layer_norm1(src)
        src1 = self.self_attn(src1, src_mask)
        src = src + self.dropout1(src1)
        src1 = self.layer_norm2(src)
        src1 = F.gelu(self.intermediate_linear1(src1))
        src1 = self.intermediate_linear2(src1)
        src1 = self.dropout(src1)
        src = src + src1
        return src


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, n_head=4, drop_out=
        0.5, d_intermediate=4)}]
