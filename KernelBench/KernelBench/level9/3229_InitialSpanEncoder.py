import torch
from torch import Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer


class InitialSpanEncoder(TransformerEncoderLayer):
    """
    The initial layer for the Segmental Transformer Encoder. Representations of
    the source sequence attend over all unmasked positions in the sequence

    The encoding at position ``i`` represents the masked span starting at
    position ``i+1``

    Args:
        src: The input sequence to encode
        attn_mask: The additive attention mask with which to mask out the
            span encoded at each position. Default: ``None``
        padding_mask: The mask for the padded positions of each key.
            Default: ``None``
    """

    def forward(self, src: 'Tensor', attn_mask: 'Tensor'=None, padding_mask:
        'Tensor'=None) ->Tensor:
        src1 = self.self_attn(src, src, src, attn_mask=attn_mask,
            key_padding_mask=padding_mask)[0]
        src = self.norm1(self.dropout1(src1))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
