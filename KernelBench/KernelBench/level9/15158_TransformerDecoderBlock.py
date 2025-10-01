import math
import torch
import torch.nn as nn


class AddAndNorm(nn.Module):

    def __init__(self, d_model):
        super(AddAndNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, residual):
        return self.layer_norm(x + residual)


class ScaledDotProductAttention(nn.Module):

    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()
        self.d_head = d_head
        self.attention_dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None):
        attention_weights = torch.matmul(q, k.transpose(-2, -1))
        scaled_attention_weights = attention_weights / math.sqrt(self.d_head)
        if mask is not None:
            scaled_attention_weights = scaled_attention_weights.masked_fill(
                mask == 0, float('-inf'))
        scaled_attention_weights = nn.functional.softmax(
            scaled_attention_weights, dim=-1)
        scaled_attention_weights = self.attention_dropout(
            scaled_attention_weights)
        weighted_v = torch.matmul(scaled_attention_weights, v)
        return weighted_v


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads
        self.dot_product_attention_layer = ScaledDotProductAttention(self.
            d_head)
        self.W_0 = nn.Linear(d_model, d_model)

    def _split_into_heads(self, q, k, v):
        q = q.view(q.size(0), q.size(1), self.n_heads, self.d_head)
        k = k.view(k.size(0), k.size(1), self.n_heads, self.d_head)
        v = v.view(v.size(0), v.size(1), self.n_heads, self.d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        return q, k, v

    def _concatenate_heads(self, attention_output):
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(attention_output.size(0),
            attention_output.size(1), -1)
        return attention_output

    def forward(self, q, k, v, mask=None):
        q, k, v = self._split_into_heads(q, k, v)
        attention_output = self.dot_product_attention_layer(q, k, v, mask)
        attention_output = self._concatenate_heads(attention_output)
        attention_output = self.W_0(attention_output)
        return attention_output


class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForwardNet, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.w_2(self.dropout(torch.relu(self.w_1(x))))


class TransformerDecoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout_proba):
        super(TransformerDecoderBlock, self).__init__()
        self.W_q_1 = nn.Linear(d_model, d_model)
        self.W_k_1 = nn.Linear(d_model, d_model)
        self.W_v_1 = nn.Linear(d_model, d_model)
        self.mha_layer_1 = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1 = nn.Dropout(dropout_proba)
        self.add_and_norm_1 = AddAndNorm(d_model)
        self.W_q_2 = nn.Linear(d_model, d_model)
        self.W_k_2 = nn.Linear(d_model, d_model)
        self.W_v_2 = nn.Linear(d_model, d_model)
        self.mha_layer_2 = MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_2 = nn.Dropout(dropout_proba)
        self.add_and_norm_2 = AddAndNorm(d_model)
        self.ffn_layer = PositionWiseFeedForwardNet(d_model, d_ff)
        self.dropout_layer_3 = nn.Dropout(dropout_proba)
        self.add_and_norm_3 = AddAndNorm(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        q_1 = self.W_q_1(x)
        k_1 = self.W_k_1(x)
        v_1 = self.W_v_1(x)
        mha_layer_1_out = self.mha_layer_1(q_1, k_1, v_1, trg_mask)
        mha_layer_1_out = self.dropout_layer_1(mha_layer_1_out)
        mha_layer_1_out = self.add_and_norm_1(mha_layer_1_out, x)
        q_2 = self.W_q_2(mha_layer_1_out)
        k_2 = self.W_k_2(encoder_output)
        v_2 = self.W_v_2(encoder_output)
        mha_layer_2_out = self.mha_layer_2(q_2, k_2, v_2, src_mask)
        mha_layer_2_out = self.dropout_layer_2(mha_layer_2_out)
        mha_layer_2_out = self.add_and_norm_2(mha_layer_2_out, mha_layer_1_out)
        ffn_out = self.ffn_layer(mha_layer_2_out)
        ffn_out = self.dropout_layer_3(ffn_out)
        ffn_out = self.add_and_norm_3(ffn_out, mha_layer_2_out)
        return ffn_out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4,
        4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'n_heads': 4, 'd_ff': 4, 'dropout_proba': 0.5}]
