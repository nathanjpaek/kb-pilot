from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.utils.checkpoint


class BertLayerNorm(nn.Module):
    """LayerNorm层, 见Transformer(一), 讲编码器(encoder)的第3部分"""

    def __init__(self, hidden_size, eps=1e-12, conditional=False):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps
        self.conditional = conditional
        if conditional is True:
            self.weight_dense = nn.Linear(2 * hidden_size, hidden_size,
                bias=False)
            self.weight_dense.weight.data.uniform_(0, 0)
            self.bias_dense = nn.Linear(2 * hidden_size, hidden_size, bias=
                False)
            self.bias_dense.weight.data.uniform_(0, 0)

    def forward(self, x):
        if self.conditional is False:
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias
        else:
            inputs = x[0]
            cond = x[1]
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(dim=1)
            weight = self.weight + self.weight_dense(cond)
            bias = self.bias + self.bias_dense(cond)
            u = inputs.mean(-1, keepdim=True)
            s = (inputs - u).pow(2).mean(-1, keepdim=True)
            x = (inputs - u) / torch.sqrt(s + self.variance_epsilon)
            return weight * x + bias


class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=config.
            layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'config': _mock_config(intermediate_size=4, hidden_size=4,
        layer_norm_eps=1, hidden_dropout_prob=0.5)}]
