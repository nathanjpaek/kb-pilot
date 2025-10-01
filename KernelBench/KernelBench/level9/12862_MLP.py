import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Create a multilayer perceptron model with variable hidden layers.
    The network will have the specified number of layers and neurons,
    with each layer using the leaky ReLU activation function. 

    Parameters
    ----------
    input_dim : int
        The number of input features.

    hidden_dims : list of int
        The number of neurons in each hidden layer. Each element
        specifies a new hidden layer.
    """

    def __init__(self, input_dim, hidden_dims, output_dim, leaky_relu_slope
        =0.2, batchnorm=False):
        """Instantiate an MLP object"""
        super(MLP, self).__init__()
        layers = hidden_dims
        if not input_dim or not isinstance(input_dim, int):
            raise ValueError("'input_dim' must be a non-zero integer.")
        if isinstance(layers, int):
            layers = [layers]
        if not all(layers) or not all(isinstance(l, int) for l in layers):
            raise ValueError(
                "'hidden_dims' must be a list of non-zero integers.")
        layers = list(layers)
        in_layers = [input_dim] + layers
        out_layers = layers + [output_dim]
        for idx, in_layer in enumerate(in_layers):
            self.add_module('linear_{}'.format(idx), nn.Linear(in_layer,
                out_layers[idx]))
            if idx < len(in_layers) - 1:
                self.add_module('leakyReLU_{}'.format(idx), nn.LeakyReLU(
                    negative_slope=leaky_relu_slope))
                if batchnorm:
                    self.add_module('BatchNorm_{}'.format(idx), nn.
                        BatchNorm1d(out_layers[idx]))

    def forward(self, x):
        """
        Run an example through the model

        Parameters
        ----------
        x : torch.Tensor
            A tensor to run through the model.

        Returns
        -------
        torch.Tensor
            The model predictions.
        """
        for idx, layer in enumerate(self._modules.values()):
            if idx < len(self._modules) - 1:
                x = layer(x)
            else:
                return layer(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'hidden_dims': 4, 'output_dim': 4}]
