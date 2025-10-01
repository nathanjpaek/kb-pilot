import torch
import torch.nn.functional as F
from torch import nn
import torch.utils.data
from torch.nn import Parameter
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class MultiheadConvAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True,
        add_bias_kv=False, add_zero_attn=False, kernel_size=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, 'embed_dim must be divisible by num_heads'
        self.scaling = self.head_dim ** -0.5
        self.kernel_size = kernel_size
        self.conv_layer_K = nn.Conv1d(embed_dim, embed_dim, self.
            kernel_size, stride=self.kernel_size)
        self.conv_layer_V = nn.Conv1d(embed_dim, embed_dim, self.
            kernel_size, stride=self.kernel_size)
        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None
        self.add_zero_attn = add_zero_attn
        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None,
        incremental_state=None, need_weights=True, static_kv=False,
        attn_mask=None):
        """Input shape: Time x Batch x Channel

        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
                if static_kv:
                    assert kv_same and not qkv_same
                    key = value = None
        else:
            saved_state = None
        if qkv_same:
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            q = self.in_proj_q(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling
        k = k.transpose(0, 1)
        k = k.transpose(1, 2)
        v = v.transpose(0, 1)
        v = v.transpose(1, 2)
        _batch_size, _d, src_len = k.size()
        size_to_add = (self.kernel_size - src_len % self.kernel_size
            ) % self.kernel_size
        k = F.pad(k, (size_to_add, 0))
        v = F.pad(v, (size_to_add, 0))
        k = self.conv_layer_K(k)
        v = self.conv_layer_V(v)
        k = k.transpose(1, 2)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        v = v.transpose(1, 2)
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(
                    attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask,
                    key_padding_mask.new_zeros(key_padding_mask.size(0), 1)
                    ], dim=1)
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim
            ).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim
                ).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim
                ).transpose(0, 1)
        if saved_state is not None:
            if 'prev_key' in saved_state:
                prev_key = saved_state['prev_key'].view(bsz * self.
                    num_heads, -1, self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
                prev_value = saved_state['prev_value'].view(bsz * self.
                    num_heads, -1, self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    v = torch.cat((prev_value, v), dim=1)
            saved_state['prev_key'] = k.view(bsz, self.num_heads, -1, self.
                head_dim)
            saved_state['prev_value'] = v.view(bsz, self.num_heads, -1,
                self.head_dim)
            self._set_input_buffer(incremental_state, saved_state)
        src_len = q.size(1)
        conv_src_len = k.size(1)
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len
            key_padding_mask = F.pad(key_padding_mask, (size_to_add, 0),
                value=1)
            key_padding_mask = key_padding_mask.view(-1, self.kernel_size)
            key_padding_mask = key_padding_mask.all(1)
            key_padding_mask = key_padding_mask.view(bsz, -1)
            assert key_padding_mask.size(1) == conv_src_len
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])],
                dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])],
                dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(
                    attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat([key_padding_mask, torch.zeros
                    (key_padding_mask.size(0), 1).type_as(key_padding_mask)
                    ], dim=1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len,
            conv_src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                conv_src_len)
            if self.onnx_trace:
                attn_weights = torch.where(key_padding_mask.unsqueeze(1).
                    unsqueeze(2), torch.Tensor([float('-Inf')]),
                    attn_weights.float()).type_as(attn_weights)
            else:
                attn_weights = attn_weights.float().masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf')
                    ).type_as(attn_weights)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                conv_src_len)
        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(
            attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=
            self.training)
        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.
            head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz,
                embed_dim)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                conv_src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None
        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer[k] = input_buffer[k].index_select(0, new_order)
            self._set_input_buffer(incremental_state, input_buffer)

    def _get_input_buffer(self, incremental_state):
        return utils.get_incremental_state(self, incremental_state,
            'attn_state') or {}

    def _set_input_buffer(self, incremental_state, buffer):
        utils.set_incremental_state(self, incremental_state, 'attn_state',
            buffer)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'embed_dim': 4, 'num_heads': 4}]
