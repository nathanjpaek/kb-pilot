from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.nn.functional as F


class BertSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.
            num_attention_heads)
        self.all_head_size = (self.num_attention_heads * self.
            attention_head_size)
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transform(self, x, linear_layer):
        bs, seq_len = x.shape[:2]
        proj = linear_layer(x)
        proj = proj.view(bs, seq_len, self.num_attention_heads, self.
            attention_head_size)
        proj = proj.transpose(1, 2)
        return proj

    def attention(self, key, query, value, attention_mask):
        S = torch.matmul(query, key.transpose(-1, -2)
            ) / self.attention_head_size ** (1 / 2)
        if attention_mask is not None:
            S += attention_mask
        S_p = F.softmax(S, dim=-1)
        S_p = self.dropout(S_p)
        context_layer = torch.matmul(S_p, value)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer

    def forward(self, hidden_states, attention_mask):
        """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
        key_layer = self.transform(hidden_states, self.key)
        value_layer = self.transform(hidden_states, self.value)
        query_layer = self.transform(hidden_states, self.query)
        attn_value = self.attention(key_layer, query_layer, value_layer,
            attention_mask)
        return attn_value


class BertLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = BertSelfAttention(config)
        self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size
            )
        self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=
            config.layer_norm_eps)
        self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.interm_dense = nn.Linear(config.hidden_size, config.
            intermediate_size)
        self.interm_af = F.gelu
        self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size
            )
        self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

    def add_norm(self, input, output, dense_layer, dropout, ln_layer):
        """
    input: the input
    output: the input that requires the sublayer to transform
    dense_layer, dropput: the sublayer
    ln_layer: layer norm that takes input+sublayer(output) #SAM NOTE Really?
    """
        output = dropout(dense_layer(output))
        output = ln_layer(input + output)
        return output

    def forward(self, hidden_states, attention_mask):
        """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
    3. a feed forward layer
    4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
    """
        attention = self.self_attention(hidden_states, attention_mask)
        x = self.add_norm(hidden_states, attention, self.attention_dense,
            self.attention_dropout, self.attention_layer_norm)
        feed_forward = self.interm_dense(x)
        feed_forward = self.interm_af(feed_forward)
        out = self.add_norm(x, feed_forward, self.out_dense, self.
            out_dropout, self.out_layer_norm)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(num_attention_heads=4, hidden_size=
        4, attention_probs_dropout_prob=0.5, layer_norm_eps=1,
        hidden_dropout_prob=0.5, intermediate_size=4)}]
