import torch
import torch.nn as nn


class Dense(nn.Module):

    def __init__(self, in_dim, out_dim, use_bias=True, activation=None,
        name=None):
        super(Dense, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bias = use_bias
        self.activation = activation
        self.name = name if name else 'dense'
        self.fc = nn.Linear(self.in_dim, self.out_dim, bias=self.use_bias)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.use_bias:
            nn.init.zeros_(self.fc.bias)

    def forward(self, inputs):
        x = self.fc(inputs)
        if self.activation:
            x = self.activation(x)
        return x


class TransformerEncoderFeedForward(nn.Module):

    def __init__(self, in_dim, out_dim, drop_out_proba, expansion_rate,
        name=None):
        super(TransformerEncoderFeedForward, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.drop_out_proba = drop_out_proba
        self.expansion_rate = expansion_rate
        self.name = name if name else 'Transformer-Encoder__Feed-Forward'
        self.hidden_dense = Dense(in_dim=self.in_dim, out_dim=self.out_dim *
            self.expansion_rate, use_bias=True, activation=nn.ReLU(), name=
            f'{self.name}__Hidden-Dense')
        self.output_dense = Dense(in_dim=self.out_dim * self.expansion_rate,
            out_dim=self.out_dim, use_bias=True, activation=None, name=
            f'{self.name}__Out-Dense')
        self.dropout = nn.Dropout(p=self.drop_out_proba)
        self.norm = nn.LayerNorm(normalized_shape=self.out_dim)

    def forward(self, inputs):
        hidden_values = self.hidden_dense(inputs)
        output = self.output_dense(hidden_values)
        output = self.dropout(output)
        return self.norm(inputs + output)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4, 'drop_out_proba': 0.5,
        'expansion_rate': 4}]
