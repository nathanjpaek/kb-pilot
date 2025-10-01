import torch
import warnings
from torch import Tensor
from torch.nn import TransformerEncoderLayer
from torch.nn.functional import *
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.activation import xavier_uniform_
from torch.nn.modules.activation import xavier_normal_
from torch.nn.modules.activation import constant_
from torch.nn.modules.activation import Parameter
from typing import Optional
import torch.utils.data.dataset
from typing import Tuple


def relational_multi_head_attention_forward(query: 'Tensor', key: 'Tensor',
    value: 'Tensor', relation: 'Tensor', embed_dim_to_check: 'int',
    num_heads: 'int', in_proj_weight: 'Tensor', in_proj_bias: 'Tensor',
    bias_k: 'Optional[Tensor]', bias_v: 'Optional[Tensor]', add_zero_attn:
    'bool', dropout_p: 'float', out_proj_weight: 'Tensor', out_proj_bias:
    'Tensor', training: 'bool'=True, key_padding_mask: 'Optional[Tensor]'=
    None, need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None,
    use_separate_proj_weight: 'bool'=False, q_proj_weight:
    'Optional[Tensor]'=None, k_proj_weight: 'Optional[Tensor]'=None,
    v_proj_weight: 'Optional[Tensor]'=None, static_k: 'Optional[Tensor]'=
    None, static_v: 'Optional[Tensor]'=None, relation_type: 'str'=None
    ) ->Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        relation: relation between queries and keys.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in different forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.


    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - relation: :math:`(L, S, N, E)` where L is the target sequence length, where S is the source sequence length,
          N is the batch size, E is the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
          will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
          3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
          S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
          positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.

        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k,
        bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(multi_head_attention_forward, tens_ops,
            query, key, value, embed_dim_to_check, num_heads,
            in_proj_weight, in_proj_bias, bias_k, bias_v, add_zero_attn,
            dropout_p, out_proj_weight, out_proj_bias, training=training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, use_separate_proj_weight=
            use_separate_proj_weight, q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight, v_proj_weight=v_proj_weight,
            static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)
    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
    scaling = float(head_dim) ** -0.5
    if not use_separate_proj_weight:
        if (query is key or torch.equal(query, key)) and (key is value or
            torch.equal(key, value)):
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3,
                dim=-1)
        elif key is value or torch.equal(key, value):
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            if key is None:
                assert value is None
                k = None
                v = None
            else:
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)
        else:
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)
        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)
        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)
        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:
                embed_dim * 2])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[embed_dim *
                2:])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling
    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, 'Only float, byte, and bool types are supported for attn_mask, not {}'.format(
            attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn(
                'Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.'
                )
            attn_mask = attn_mask
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError(
                    'The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0),
                key.size(0)]:
                raise RuntimeError(
                    'The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".
                format(attn_mask.dim()))
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn(
            'Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.'
            )
        key_padding_mask = key_padding_mask
    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, 'bias cannot be added to static key.'
            assert static_v is None, 'bias cannot be added to static value.'
    else:
        assert bias_k is None
        assert bias_v is None
    r = None
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k
    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v
    src_len = k.size(1)
    if relation is not None:
        if relation_type == 'qk+r':
            r = relation.contiguous().view(tgt_len, -1, bsz * num_heads, 1
                ).squeeze(3).permute(2, 0, 1)
        elif relation_type == 'q(k+r)':
            r = relation.contiguous().view(tgt_len, src_len, bsz *
                num_heads, head_dim).permute(2, 0, 1)
            r = r.view(bsz * num_heads, tgt_len * src_len, head_dim)
            r = torch.bmm(r, q.unsqueeze(2).repeat(1, 1, src_len, 1).view(
                bsz * num_heads, tgt_len * src_len, head_dim).transpose(1, 2)
                ).view(bsz * num_heads, tgt_len, src_len)
    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len
    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=
            k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=
            v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    if r is not None:
        attn_output_weights = attn_output_weights + r
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len,
        src_len]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask
    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads,
            tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(key_padding_mask
            .unsqueeze(1).unsqueeze(2), float('-inf'))
        attn_output_weights = attn_output_weights.view(bsz * num_heads,
            tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p,
        training=training)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len,
        bsz, embed_dim)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        attn_output_weights = attn_output_weights.view(bsz, num_heads,
            tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class RelationalMultiheadAttention(MultiheadAttention):
    """Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \\text{MultiHead}(Q, K, V) = \\text{Concat}(head_1,\\dots,head_h)W^O

    where :math:`head_i = \\text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in value. Default: None.

    Note that if :attr:`kdim` and :attr:`vdim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.

    Examples::

        >>> realational_multihead_attn = RelationalMultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = realational_multihead_attn(query, key, value)
    """
    bias_k: 'Optional[torch.Tensor]'
    bias_v: 'Optional[torch.Tensor]'

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
        add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None,
        add_relation=False, rdim=None, relation_type=None):
        super(RelationalMultiheadAttention, self).__init__(embed_dim=
            embed_dim, num_heads=num_heads, dropout=dropout, bias=bias,
            add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn, kdim=kdim,
            vdim=vdim)
        self.add_relation = add_relation
        self.rdim = rdim if rdim is not None else embed_dim
        self.relation_type = relation_type if relation_type else 'qk+r'
        if self.add_relation:
            if relation_type == 'qk+r':
                self.r_proj_weight = Parameter(torch.Tensor(num_heads, self
                    .rdim))
                self.r_proj_bias = Parameter(torch.empty(num_heads))
            elif relation_type == 'q(k+r)':
                self.r_proj_weight = Parameter(torch.Tensor(embed_dim, self
                    .rdim))
                self.r_proj_bias = Parameter(torch.empty(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
        if hasattr(self, 'add_relation') and self.add_relation:
            xavier_uniform_(self.r_proj_weight)
            constant_(self.r_proj_bias, 0.0)
        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True
        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: 'Tensor', key: 'Tensor', value: 'Tensor',
        relation_dict=None, key_padding_mask: 'Optional[Tensor]'=None,
        need_weights: 'bool'=True, attn_mask: 'Optional[Tensor]'=None) ->Tuple[
        Tensor, Optional[Tensor]]:
        """
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. When given a binary mask and a value is True,
            the corresponding value on the attention layer will be ignored. When given
            a byte mask and a value is non-zero, the corresponding value on the attention
            layer will be ignored
        need_weights: output attn_output_weights.
        attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
            the batches while a 3D mask allows to specify a different mask for the entries of each batch.

    Shapes for inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
          If a ByteTensor is provided, the non-zero positions will be ignored while the position
          with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
          value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
          source sequence length.

          If a 3D mask: :math:`(N\\cdot\\text{num\\_heads}, L, S)` where N is the batch size, L is the target sequence
          length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
          the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
          while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
          is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
          is provided, it will be added to the attention weight.

    Shapes for outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if relation_dict is not None:
            relation_labels = relation_dict['relation_labels']
            relation_ids = relation_dict['relation_ids']
            batch_index = relation_dict['batch_index']
            pad_embedding = relation_dict['pad_embedding']
            relation_labels = linear(relation_labels, self.r_proj_weight,
                self.r_proj_bias)
            pad_embedding = linear(pad_embedding.unsqueeze(0), self.
                r_proj_weight, self.r_proj_bias).squeeze()
            tgt_length, bsz, _ = query.size()
            src_length, _, _ = key.size()
            relation = pad_embedding.view(1, 1, 1, -1).repeat(bsz,
                tgt_length, src_length, 1)
            relation[batch_index, relation_ids[:, :, 0], relation_ids[:, :, 1]
                ] = relation_labels
            relation = relation.permute(1, 2, 0, 3)
            if not self._qkv_same_embed_dim:
                return relational_multi_head_attention_forward(query, key,
                    value, relation, self.embed_dim, self.num_heads, self.
                    in_proj_weight, self.in_proj_bias, self.bias_k, self.
                    bias_v, self.add_zero_attn, self.dropout, self.out_proj
                    .weight, self.out_proj.bias, training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=
                    need_weights, attn_mask=attn_mask,
                    use_separate_proj_weight=True, q_proj_weight=self.
                    q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
            else:
                return relational_multi_head_attention_forward(query, key,
                    value, relation, self.embed_dim, self.num_heads, self.
                    in_proj_weight, self.in_proj_bias, self.bias_k, self.
                    bias_v, self.add_zero_attn, self.dropout, self.out_proj
                    .weight, self.out_proj.bias, training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=
                    need_weights, attn_mask=attn_mask)
        elif not self._qkv_same_embed_dim:
            return relational_multi_head_attention_forward(query, key,
                value, None, self.embed_dim, self.num_heads, self.
                in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                self.add_zero_attn, self.dropout, self.out_proj.weight,
                self.out_proj.bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=
                need_weights, attn_mask=attn_mask, use_separate_proj_weight
                =True, q_proj_weight=self.q_proj_weight, k_proj_weight=self
                .k_proj_weight, v_proj_weight=self.v_proj_weight)
        else:
            return relational_multi_head_attention_forward(query, key,
                value, None, self.embed_dim, self.num_heads, self.
                in_proj_weight, self.in_proj_bias, self.bias_k, self.bias_v,
                self.add_zero_attn, self.dropout, self.out_proj.weight,
                self.out_proj.bias, training=self.training,
                key_padding_mask=key_padding_mask, need_weights=
                need_weights, attn_mask=attn_mask)


class RelationalTransformerEncoderLayer(TransformerEncoderLayer):
    """TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> encoder_layer = RelationalTransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> rel = torch.rand(10, 10, 32, 512)
        >>> out = encoder_layer(src, rel)
    """

    def __init__(self, d_model, nhead, add_relation=False, dim_feedforward=
        2048, dropout=0.1, activation='relu', relation_type=None):
        super(RelationalTransformerEncoderLayer, self).__init__(d_model,
            nhead, dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation)
        self.self_attn = RelationalMultiheadAttention(d_model, nhead,
            add_relation=add_relation, dropout=dropout, relation_type=
            relation_type)

    def forward(self, src: 'Tensor', relation=None, src_mask:
        'Optional[Tensor]'=None, src_key_padding_mask: 'Optional[Tensor]'=None
        ) ->Tensor:
        """Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        src2 = self.self_attn(src, src, src, relation_dict=relation,
            attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'nhead': 4}]
