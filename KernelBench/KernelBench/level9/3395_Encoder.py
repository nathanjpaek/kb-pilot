import torch
import torch.nn as nn
import torch.nn.functional as F


class Lambda(nn.Module):
    """An easy way to create a pytorch layer for a simple `func`."""

    def __init__(self, func):
        """create a layer that simply calls `func` with `x`"""
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class FFN(nn.Module):
    """
    Feed-Forward Network
    """

    def __init__(self, d_inner_hid, d_model, dropout_rate):
        super(FFN, self).__init__()
        self.dropout_rate = dropout_rate
        self.fc1 = torch.nn.Linear(in_features=d_model, out_features=
            d_inner_hid)
        self.fc2 = torch.nn.Linear(in_features=d_inner_hid, out_features=
            d_model)

    def forward(self, x):
        hidden = self.fc1(x)
        hidden = F.relu(hidden)
        if self.dropout_rate:
            hidden = F.dropout(hidden, p=self.dropout_rate)
        out = self.fc2(hidden)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention
    """

    def __init__(self, d_key, d_value, d_model, n_head=1, dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_key = d_key
        self.d_value = d_value
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.q_fc = torch.nn.Linear(in_features=d_model, out_features=d_key *
            n_head, bias=False)
        self.k_fc = torch.nn.Linear(in_features=d_model, out_features=d_key *
            n_head, bias=False)
        self.v_fc = torch.nn.Linear(in_features=d_model, out_features=
            d_value * n_head, bias=False)
        self.proj_fc = torch.nn.Linear(in_features=d_value * n_head,
            out_features=d_model, bias=False)

    def _prepare_qkv(self, queries, keys, values, cache=None):
        if keys is None:
            keys, values = queries, queries
            static_kv = False
        else:
            static_kv = True
        q = self.q_fc(queries)
        q = torch.reshape(q, shape=[q.size(0), q.size(1), self.n_head, self
            .d_key])
        q = q.permute(0, 2, 1, 3)
        if cache is not None and static_kv and 'static_k' in cache:
            k = cache['static_k']
            v = cache['static_v']
        else:
            k = self.k_fc(keys)
            v = self.v_fc(values)
            k = torch.reshape(k, shape=[k.size(0), k.size(1), self.n_head,
                self.d_key])
            k = k.permute(0, 2, 1, 3)
            v = torch.reshape(v, shape=[v.size(0), v.size(1), self.n_head,
                self.d_value])
            v = v.permute(0, 2, 1, 3)
        if cache is not None:
            if static_kv and 'static_k' not in cache:
                cache['static_k'], cache['static_v'] = k, v
            elif not static_kv:
                cache_k, cache_v = cache['k'], cache['v']
                k = torch.cat([cache_k, k], dim=2)
                v = torch.cat([cache_v, v], dim=2)
                cache['k'], cache['v'] = k, v
        return q, k, v

    def forward(self, queries, keys, values, attn_bias, cache=None):
        keys = queries if keys is None else keys
        values = keys if values is None else values
        q, k, v = self._prepare_qkv(queries, keys, values, cache)
        product = torch.matmul(q, k.transpose(2, 3))
        product = product * self.d_model ** -0.5
        if attn_bias is not None:
            product += attn_bias
        weights = F.softmax(product, dim=-1)
        if self.dropout_rate:
            weights = F.dropout(weights, p=self.dropout_rate)
        out = torch.matmul(weights, v)
        out = out.permute(0, 2, 1, 3)
        out = torch.reshape(out, shape=[out.size(0), out.size(1), out.shape
            [2] * out.shape[3]])
        out = self.proj_fc(out)
        return out


class LambdaXY(nn.Module):
    """An easy way to create a pytorch layer for a simple `func`."""

    def __init__(self, func):
        """create a layer that simply calls `func` with `x`"""
        super().__init__()
        self.func = func

    def forward(self, x, y):
        return self.func(x, y)


class PrePostProcessLayer(nn.Module):
    """
    PrePostProcessLayer
    """

    def __init__(self, process_cmd, d_model, dropout_rate):
        super(PrePostProcessLayer, self).__init__()
        self.process_cmd = process_cmd
        self.functors = nn.ModuleList()
        cur_a_len = 0
        cur_n_len = 0
        cur_d_len = 0
        for cmd in self.process_cmd:
            if cmd == 'a':
                self.functors.add_module('add_res_connect_{}'.format(
                    cur_a_len), LambdaXY(lambda x, y: x + y if y is not
                    None else x))
                cur_a_len += 1
            elif cmd == 'n':
                layerNorm = torch.nn.LayerNorm(normalized_shape=d_model,
                    elementwise_affine=True, eps=1e-05)
                self.functors.add_module('layer_norm_%d' % cur_n_len, layerNorm
                    )
                cur_n_len += 1
            elif cmd == 'd':
                self.functors.add_module('add_drop_{}'.format(cur_d_len),
                    Lambda(lambda x: F.dropout(x, p=dropout_rate) if
                    dropout_rate else x))
                cur_d_len += 1

    def forward(self, x, residual=None):
        for i, (cmd, functor) in enumerate(zip(self.process_cmd, self.functors)
            ):
            if cmd == 'a':
                x = functor(x, residual)
            else:
                x = functor(x)
        return x


class EncoderLayer(nn.Module):
    """
    EncoderLayer
    """

    def __init__(self, n_head, d_key, d_value, d_model, d_inner_hid,
        prepostprocess_dropout, attention_dropout, relu_dropout,
        preprocess_cmd='n', postprocess_cmd='da'):
        super(EncoderLayer, self).__init__()
        self.preprocesser1 = PrePostProcessLayer(preprocess_cmd, d_model,
            prepostprocess_dropout)
        self.self_attn = MultiHeadAttention(d_key, d_value, d_model, n_head,
            attention_dropout)
        self.postprocesser1 = PrePostProcessLayer(postprocess_cmd, d_model,
            prepostprocess_dropout)
        self.preprocesser2 = PrePostProcessLayer(preprocess_cmd, d_model,
            prepostprocess_dropout)
        self.ffn = FFN(d_inner_hid, d_model, relu_dropout)
        self.postprocesser2 = PrePostProcessLayer(postprocess_cmd, d_model,
            prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        attn_output = self.self_attn(self.preprocesser1(enc_input), None,
            None, attn_bias)
        attn_output = self.postprocesser1(attn_output, enc_input)
        ffn_output = self.ffn(self.preprocesser2(attn_output))
        ffn_output = self.postprocesser2(ffn_output, attn_output)
        return ffn_output


class Encoder(nn.Module):
    """
    encoder
    """

    def __init__(self, n_layer, n_head, d_key, d_value, d_model,
        d_inner_hid, prepostprocess_dropout, attention_dropout,
        relu_dropout, preprocess_cmd='n', postprocess_cmd='da'):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        for i in range(n_layer):
            encoderLayer = EncoderLayer(n_head, d_key, d_value, d_model,
                d_inner_hid, prepostprocess_dropout, attention_dropout,
                relu_dropout, preprocess_cmd, postprocess_cmd)
            self.encoder_layers.add_module('layer_%d' % i, encoderLayer)
        self.processer = PrePostProcessLayer(preprocess_cmd, d_model,
            prepostprocess_dropout)

    def forward(self, enc_input, attn_bias):
        for encoder_layer in self.encoder_layers:
            enc_output = encoder_layer(enc_input, attn_bias)
            enc_input = enc_output
        enc_output = self.processer(enc_output)
        return enc_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'n_layer': 1, 'n_head': 4, 'd_key': 4, 'd_value': 4,
        'd_model': 4, 'd_inner_hid': 4, 'prepostprocess_dropout': 0.5,
        'attention_dropout': 0.5, 'relu_dropout': 0.5}]
