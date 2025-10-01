import torch
from typing import List
from collections import OrderedDict
from torch import nn


def is_dropout_module(module: 'nn.Module', dropout_modules:
    'List[nn.Module]'=[nn.Dropout, nn.Dropout2d, nn.Dropout3d]) ->bool:
    """Detect if a PyTorch Module is a Dropout layer
    module (nn.Module): Module to check
    dropout_modules (List[nn.Module], optional): List of Modules that count as Dropout layers.
    RETURNS (bool): True if module is a Dropout layer.
    """
    for m in dropout_modules:
        if isinstance(module, m):
            return True
    return False


class TorchEntityRecognizer(nn.Module):
    """Torch Entity Recognizer Model Head"""

    def __init__(self, nI: 'int', nH: 'int', nO: 'int', dropout: 'float'):
        """Initialize TorchEntityRecognizer.
        nI (int): Input Dimension
        nH (int): Hidden Dimension Width
        nO (int): Output Dimension Width
        dropout (float): Dropout ratio (0 - 1.0)
        """
        super(TorchEntityRecognizer, self).__init__()
        nI = nI or 1
        nO = nO or 1
        self.nH = nH
        self.model = nn.Sequential(OrderedDict({'input_layer': nn.Linear(nI,
            nH), 'input_activation': nn.ReLU(), 'input_dropout': nn.
            Dropout2d(dropout), 'output_layer': nn.Linear(nH, nO),
            'output_dropout': nn.Dropout2d(dropout), 'softmax': nn.Softmax(
            dim=1)}))

    def forward(self, inputs: 'torch.Tensor') ->torch.Tensor:
        """Forward pass of the model.
        inputs (torch.Tensor): Batch of outputs from spaCy tok2vec layer
        RETURNS (torch.Tensor): Batch of results with a score for each tag for each token
        """
        return self.model(inputs)

    def _set_layer_shape(self, name: 'str', nI: 'int', nO: 'int'):
        """Dynamically set the shape of a layer
        name (str): Layer name
        nI (int): New input shape
        nO (int): New output shape
        """
        with torch.no_grad():
            layer = getattr(self.model, name)
            layer.out_features = nO
            layer.weight = nn.Parameter(torch.Tensor(nO, nI))
            if layer.bias is not None:
                layer.bias = nn.Parameter(torch.Tensor(nO))
            layer.reset_parameters()

    def set_input_shape(self, nI: 'int'):
        """Dynamically set the shape of the input layer
        nI (int): New input layer shape
        """
        self._set_layer_shape('input_layer', nI, self.nH)

    def set_output_shape(self, nO: 'int'):
        """Dynamically set the shape of the output layer
        nO (int): New output layer shape
        """
        self._set_layer_shape('output_layer', self.nH, nO)

    def set_dropout_rate(self, dropout: 'float'):
        """Set the dropout rate of all Dropout layers in the model.
        dropout (float): Dropout rate to set
        """
        dropout_layers = [module for module in self.modules() if
            is_dropout_module(module)]
        for layer in dropout_layers:
            layer.p = dropout


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nI': 4, 'nH': 4, 'nO': 4, 'dropout': 0.5}]
