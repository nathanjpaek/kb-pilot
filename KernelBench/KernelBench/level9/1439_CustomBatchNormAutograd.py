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
        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        self.eps = eps

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
        shape = input.shape
        if len(shape) == 1:
            input = input.unsqueeze(0)
            shape = input.shape
        elif len(shape) > 2:
            raise ValueError(
                f'Expected 2D input. Instead, got {len(shape)}D input with shape of {shape}.'
                )
        elif shape[1] != self.gamma.shape[0]:
            raise ValueError(
                f'Expected input of shape batch_size x {self.gamma.shape[0]}. Instead, got input withshape of {shape}.'
                )
        mean = input.mean(0)
        var = input.var(0)
        x_hat = (input - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * x_hat + self.beta
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_neurons': 4}]
