import torch
import torch.nn as nn


class CustomBatchNormAutograd(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    The operations called in self.forward track the history if the input tensors have the
    flag requires_grad set to True. The backward pass does not need to be implemented, it
    is dealt with by the automatic differentiation provided by PyTorch.
    """

    def __init__(self, n_neurons, eps=1e-05):
        """
        Initializes CustomBatchNormAutograd object.
        
        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        
        TODO:
          Save parameters for the number of neurons and eps.
          Initialize parameters gamma and beta via nn.Parameter
        """
        super(CustomBatchNormAutograd, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.gammas = torch.nn.Parameter(torch.ones(n_neurons))
        self.betas = torch.nn.Parameter(torch.zeros(n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization
        
        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        
        TODO:
          Check for the correctness of the shape of the input tensor.
          Implement batch normalization forward pass as given in the assignment.
          For the case that you make use of torch.var be aware that the flag unbiased=False should be set.
        """
        shape = input.size()
        if len(shape) == 1:
            input.view(1, -1)
            shape = input.size()
        if len(shape) > 2:
            raise ValueError('Expected 1-D or 2-D tensor (got {})'.format(
                str(shape)))
        elif input.shape[1] != self.n_neurons:
            raise ValueError('Expected _ x {} tensor (got {} x {})'.format(
                str(self.n_neurons), str(shape[0]), str(shape[1])))
        x = input
        mu = x.mean(dim=0)
        std = x.std(dim=0)
        x_hat = (x - mu) / torch.sqrt(std * std + self.eps)
        out = self.gammas * x_hat + self.betas
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_neurons': 4}]
