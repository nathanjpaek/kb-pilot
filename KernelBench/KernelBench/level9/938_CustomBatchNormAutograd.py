import torch
import torch.nn as nn


class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True.
    """

    def __init__(self, n_neurons, eps=1e-05):
        """
        Initializes CustomBatchNormAutograd object.
    
        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability

        """
        super(CustomBatchNormAutograd, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor

        """
        batch_size = input.shape[0]
        assert input.shape[1
            ] == self.n_neurons, 'Input not in the correct shape'
        mean = 1 / batch_size * torch.sum(input, dim=0)
        var = input.var(dim=0, unbiased=False)
        norm = (input - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * norm + self.beta
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_neurons': 4}]
