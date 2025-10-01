import torch
import torch.nn as nn
import torch.nn.functional as F


def transformer_weights_init(module, std_init_range=0.02, xavier=True):
    """
    Initialize different weights in Transformer model.
    Args:
        module: torch.nn.Module to be initialized
        std_init_range: standard deviation of normal initializer
        xavier: if True, xavier initializer will be used in Linear layers
            as was proposed in AIAYN paper, otherwise normal initializer
            will be used (like in BERT paper)
    """
    if isinstance(module, nn.Linear):
        if xavier:
            nn.init.xavier_uniform_(module.weight)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std_init_range)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class SelfAttention(nn.Module):

    def __init__(self, hidden_size, batch_first=True):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.register_parameter('att_weights', nn.Parameter(torch.Tensor(1,
            hidden_size), requires_grad=True))
        nn.init.xavier_uniform_(self.att_weights.data)

    def get_mask(self):
        pass

    def forward(self, hidden_states, attention_mask=None):
        if self.batch_first:
            batch_size, _max_len = hidden_states.size()[:2]
        else:
            _max_len, batch_size = hidden_states.size()[:2]
        weights = torch.bmm(hidden_states, self.att_weights.permute(1, 0).
            unsqueeze(0).repeat(batch_size, 1, 1))
        attentions = F.softmax(torch.tanh(weights.squeeze()), dim=-1)
        masked = attentions * attention_mask
        if len(attentions.shape) == 1:
            attentions = attentions.unsqueeze(0)
        _sums = masked.sum(-1, keepdim=True).expand(attentions.shape)
        attentions = masked.div(_sums)
        weighted = torch.mul(hidden_states, attentions.unsqueeze(-1).
            expand_as(hidden_states))
        representations = weighted.sum(1).squeeze(dim=1)
        return representations, attentions


class MultiLayerPerceptron(torch.nn.Module):
    """
    A simple MLP that can either be used independently or put on top
    of pretrained models (such as BERT) and act as a classifier.
    Args:
        hidden_size (int): the size of each layer
        num_classes (int): number of output classes
        num_layers (int): number of layers
        activation (str): type of activations for layers in between
        log_softmax (bool): whether to add a log_softmax layer before output
    """

    def __init__(self, hidden_size: 'int', num_classes: 'int', num_layers:
        'int'=2, activation: 'str'='relu', log_softmax: 'bool'=True):
        super().__init__()
        self.layers = 0
        activations = {'relu': nn.ReLU(), 'gelu': nn.GELU(), 'sigmoid': nn.
            Sigmoid(), 'tanh': nn.Tanh()}
        for _ in range(num_layers - 1):
            layer = torch.nn.Linear(hidden_size, hidden_size)
            setattr(self, f'layer{self.layers}', layer)
            setattr(self, f'layer{self.layers + 1}', activations[activation])
            self.layers += 2
        layer = torch.nn.Linear(hidden_size, num_classes)
        setattr(self, f'layer{self.layers}', layer)
        self.layers += 1
        self.log_softmax = log_softmax

    @property
    def last_linear_layer(self):
        return getattr(self, f'layer{self.layers - 1}')

    def forward(self, hidden_states):
        output_states = hidden_states[:]
        for i in range(self.layers):
            output_states = getattr(self, f'layer{i}')(output_states)
        if self.log_softmax:
            output_states = torch.log_softmax(output_states, dim=-1)
        else:
            output_states = torch.softmax(output_states, dim=-1)
        return output_states


class SequenceClassifier(nn.Module):

    def __init__(self, hidden_size: 'int', num_classes: 'int', num_layers:
        'int'=2, activation: 'str'='relu', log_softmax: 'bool'=True,
        dropout: 'float'=0.0, use_transformer_init: 'bool'=True, pooling:
        'str'='mean', idx_conditioned_on: 'int'=None):
        """
        Initializes the SequenceClassifier module.
        Args:
            hidden_size: the hidden size of the mlp head on the top of the encoder
            num_classes: number of the classes to predict
            num_layers: number of the linear layers of the mlp head on the top of the encoder
            activation: type of activations between layers of the mlp head
            log_softmax: applies the log softmax on the output
            dropout: the dropout used for the mlp head
            use_transformer_init: initializes the weights with the same approach used in Transformer
            idx_conditioned_on: index of the token to use as the sequence representation for the classification task, default is the first token
        """
        super().__init__()
        self.log_softmax = log_softmax
        self._idx_conditioned_on = idx_conditioned_on
        self.pooling = pooling
        self.mlp = MultiLayerPerceptron(hidden_size=hidden_size * 2 if 
            pooling == 'mean_max' else hidden_size, num_classes=num_classes,
            num_layers=num_layers, activation=activation, log_softmax=
            log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module,
                xavier=False))
        if pooling == 'attention':
            self.attention = SelfAttention(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        hidden_states = self.dropout(hidden_states)
        if self.pooling == 'token':
            pooled = hidden_states[:, self._idx_conditioned_on]
        elif self.pooling == 'attention':
            pooled, _att = self.attention(hidden_states, attention_mask)
        else:
            if attention_mask is None:
                ct = hidden_states.shape[1]
            else:
                hidden_states = hidden_states * attention_mask.unsqueeze(2)
                ct = torch.sum(attention_mask, axis=1).unsqueeze(1)
            pooled_sum = torch.sum(hidden_states, axis=1)
            if self.pooling == 'mean' or self.pooling == 'mean_max':
                pooled_mean = torch.div(pooled_sum, ct)
            if self.pooling == 'max' or self.pooling == 'mean_max':
                pooled_max = torch.max(hidden_states, axis=1)[0]
            pooled = (pooled_mean if self.pooling == 'mean' else pooled_max if
                self.pooling == 'max' else torch.cat([pooled_mean,
                pooled_max], axis=-1))
        logits = self.mlp(pooled)
        return logits, pooled


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_classes': 4}]
