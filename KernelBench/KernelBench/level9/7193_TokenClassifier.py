import torch
import torch.nn as nn


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


class TokenClassifier(nn.Module):
    """
    A module to perform token level classification tasks such as Named entity recognition.
    """

    def __init__(self, hidden_size: 'int', num_classes: 'int', num_layers:
        'int'=1, activation: 'str'='relu', log_softmax: 'bool'=True,
        dropout: 'float'=0.0, use_transformer_init: 'bool'=True) ->None:
        """
        Initializes the Token Classifier module.
        Args:
            hidden_size: the size of the hidden dimension
            num_classes: number of classes
            num_layers: number of fully connected layers in the multilayer perceptron (MLP)
            activation: activation to usee between fully connected layers in the MLP
            log_softmax: whether to apply softmax to the output of the MLP
            dropout: dropout to apply to the input hidden states
            use_transformer_init: whether to initialize the weights of the classifier head with the same approach used in Transformer
        """
        super().__init__()
        self.log_softmax = log_softmax
        self.mlp = MultiLayerPerceptron(hidden_size, num_classes,
            num_layers=num_layers, activation=activation, log_softmax=
            log_softmax)
        self.dropout = nn.Dropout(dropout)
        if use_transformer_init:
            self.apply(lambda module: transformer_weights_init(module,
                xavier=False))

    def forward(self, hidden_states):
        """
        Performs the forward step of the module.
        Args:
            hidden_states: batch of hidden states (for example, from the BERT encoder module)
                [BATCH_SIZE x SEQ_LENGTH x HIDDEN_SIZE]
        Returns: logits value for each class [BATCH_SIZE x SEQ_LENGTH x NUM_CLASSES]
        """
        hidden_states = self.dropout(hidden_states)
        logits = self.mlp(hidden_states)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_classes': 4}]
