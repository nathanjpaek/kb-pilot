import torch
from torch import Tensor
from torch.nn.modules.transformer import TransformerEncoderLayer


class SubsequentSpanEncoder(TransformerEncoderLayer):
    """
    The subsequent layers for the Segmental Transformer Encoder. The encoded
    representations from previous layers attend over all unmasked positions of
    the original source sequence (to prevent information leaks from "under" the
    mask)

    The encoding at position ``i`` represents the masked span starting at
    position ``i+1``

    Args:
        enc: The encoded representation from previous segmental encoder layers
        src: The original input sequence to encode
        attn_mask: The additive attention mask with which to mask out the
            span encoded at each position. Default: ``None``
        padding_mask: The mask for the padded positions of each key.
            Default: ``None``
    """

    def forward(self, enc: 'Tensor', src: 'Tensor', attn_mask: 'Tensor'=
        None, padding_mask: 'Tensor'=None) ->Tensor:
        enc1 = self.self_attn(enc, src, src, attn_mask=attn_mask,
            key_padding_mask=padding_mask)[0]
        enc = self.norm1(enc + self.dropout1(enc1))
        enc2 = self.linear2(self.dropout(self.activation(self.linear1(enc))))
        enc = self.norm2(enc + self.dropout2(enc2))
        return enc


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
