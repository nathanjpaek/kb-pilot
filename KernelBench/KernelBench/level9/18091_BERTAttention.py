from _paritybench_helpers import _mock_config
import copy
import math
import torch
import torch.nn as nn


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BERTLayerNorm(nn.Module):

    def __init__(self, config, multi_params=None, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        if multi_params is not None:
            self.gamma = nn.Parameter(torch.ones(config.hidden_size_aug))
            self.beta = nn.Parameter(torch.zeros(config.hidden_size_aug))
        else:
            self.gamma = nn.Parameter(torch.ones(config.hidden_size))
            self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class BERTSelfAttention(nn.Module):

    def __init__(self, config, multi_params=None):
        super(BERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention heads (%d)'
                 % (config.hidden_size, config.num_attention_heads))
        if multi_params is not None:
            self.num_attention_heads = multi_params
            self.attention_head_size = int(config.hidden_size_aug / self.
                num_attention_heads)
            self.all_head_size = (self.num_attention_heads * self.
                attention_head_size)
            hidden_size = config.hidden_size_aug
        else:
            self.num_attention_heads = config.num_attention_heads
            self.attention_head_size = int(config.hidden_size / config.
                num_attention_heads)
            self.all_head_size = (self.num_attention_heads * self.
                attention_head_size)
            hidden_size = config.hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.
            attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1,
            -2))
        attention_scores = attention_scores / math.sqrt(self.
            attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.
            all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class AdapterLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.adapter_linear1 = nn.Linear(config.hidden_size, config.
            adapter_size)
        self.gelu = gelu
        self.adapter_linear2 = nn.Linear(config.adapter_size, config.
            hidden_size)

    def forward(self, input_tensor):
        net = self.adapter_linear1(input_tensor)
        net = self.gelu(net)
        net = self.adapter_linear2(net)
        return net + input_tensor


class BERTLowRank(nn.Module):

    def __init__(self, config, extra_dim=None):
        super(BERTLowRank, self).__init__()
        if config.extra_dim:
            self.aug_dense = nn.Linear(config.hidden_size, config.extra_dim)
            self.aug_dense2 = nn.Linear(config.extra_dim, config.hidden_size)
        else:
            self.aug_dense = nn.Linear(config.hidden_size, config.
                hidden_size_aug)
            self.aug_dense2 = nn.Linear(config.hidden_size_aug, config.
                hidden_size)
        self.config = config
        self.hidden_act_fn = gelu

    def forward(self, hidden_states, attention_mask=None):
        hidden_states_aug = self.aug_dense(hidden_states)
        hidden_states_aug = self.hidden_act_fn(hidden_states_aug)
        hidden_states = self.aug_dense2(hidden_states_aug)
        return hidden_states


class BERTSelfOutput(nn.Module):

    def __init__(self, config, multi_params=None, houlsby=False):
        super(BERTSelfOutput, self).__init__()
        if houlsby:
            multi = BERTLowRank(config)
            self.multi_layers = nn.ModuleList([copy.deepcopy(multi) for _ in
                range(config.num_tasks)])
        if multi_params is not None:
            self.dense = nn.Linear(config.hidden_size_aug, config.
                hidden_size_aug)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if config.adapter == 'adapter_google':
            adapter = AdapterLayer(config)
            self.adapters = nn.ModuleList([copy.deepcopy(adapter) for _ in
                range(config.num_tasks)])
        self.LayerNorm = BERTLayerNorm(config, multi_params)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.houlsby = houlsby
        self.adapter = config.adapter

    def forward(self, hidden_states, input_tensor, attention_mask=None, i=0):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if self.houlsby:
            hidden_states = hidden_states + self.multi_layers[i](hidden_states,
                attention_mask)
        if self.adapter == 'adapter_google':
            hidden_states = self.adapters[i](hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BERTAttention(nn.Module):

    def __init__(self, config, multi_params=None, houlsby=False):
        super(BERTAttention, self).__init__()
        self.self = BERTSelfAttention(config, multi_params)
        self.output = BERTSelfOutput(config, multi_params, houlsby)

    def forward(self, input_tensor, attention_mask, i=0):
        self_output = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor,
            attention_mask, i=i)
        return attention_output


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(hidden_size=4, num_attention_heads=
        4, attention_probs_dropout_prob=0.5, adapter=4, hidden_dropout_prob
        =0.5)}]
