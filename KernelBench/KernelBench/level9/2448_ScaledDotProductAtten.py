import torch
import numpy as np
from torch import nn


class ScaledDotProductAtten(nn.Module):
    """
    Scaled dot-product attention mechainsm

    公式：
        $  Attention(Q, K, V) = softmax(rac{Q K^T}{\\sqrt{d_k}})*V $

    ![](https://raw.githubusercontent.com/LinXueyuanStdio/scRNN-seq/master/art/2.png)
    """

    def __init__(self, encode_size, atten_dropout=0.1):
        super(ScaledDotProductAtten, self).__init__()
        encode_size = 2 * encode_size
        self.scale = encode_size ** -0.5
        self.dropout = nn.Dropout(atten_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, atten_mask=None):
        """
        前向传播.

        Args:
                Q: Queries，[B, L_q, D_q]
                K: Keys，[B, L_k, D_k]
                V: Values，[B, L_v, D_v]，一般来说就是k
                scale: 缩放因子，一个浮点标量
                attn_mask: Masking，[B, L_q, L_k]
        Returns:
                上下文张量和attetention张量
        """
        atten = torch.bmm(query, key.transpose(1, 2)) * self.scale
        if atten_mask:
            atten.masked_fill_(atten_mask, -np.inf)
        atten = self.softmax(atten)
        atten = self.dropout(atten)
        context = torch.bmm(atten, value)
        return context, atten


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'encode_size': 4}]
