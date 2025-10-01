import torch
import torch.nn as nn


class CustomBatchNormManualFunction(torch.autograd.Function):
    """
    This torch.autograd.Function implements a functional custom version of the batch norm operation for MLPs.
    Using torch.autograd.Function allows you to write a custom backward function.
    The function will be called from the nn.Module CustomBatchNormManualModule
    Inside forward the tensors are (automatically) not recorded for automatic differentiation since the backward
    pass is done via the backward method.
    The forward pass is not called directly but via the apply() method. This makes sure that the context objects
    are dealt with correctly. Example:
    my_bn_fct = CustomBatchNormManualFunction()
    normalized = fct.apply(input, gamma, beta, eps)
    """

    @staticmethod
    def forward(ctx, input, gamma, beta, eps=1e-05):
        """
        Compute the batch normalization

        Args:
        ctx: context object handling storing and retrival of tensors and constants and specifying
             whether tensors need gradients in backward pass
        input: input tensor of shape (n_batch, n_neurons)
        gamma: variance scaling tensor, applied per neuron, shpae (n_neurons)
        beta: mean bias tensor, applied per neuron, shpae (n_neurons)
        eps: small float added to the variance for stability
        Returns:
        out: batch-normalized tensor
        """
        batch_size = input.shape[0]
        mean = 1 / batch_size * torch.sum(input, dim=0)
        var = input.var(dim=0, unbiased=False)
        norm = (input - mean) / torch.sqrt(var + eps)
        out = gamma * norm + beta
        ctx.save_for_backward(norm, gamma, var)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
    Compute backward pass of the batch normalization.
    
    Args:
      ctx: context object handling storing and retrival of tensors and constants and specifying
           whether tensors need gradients in backward pass
    Returns:
      out: tuple containing gradients for all input arguments

    """
        normalized, gamma, var = ctx.saved_tensors
        eps = ctx.eps
        B = grad_output.shape[0]
        grad_gamma = (grad_output * normalized).sum(0)
        grad_beta = torch.sum(grad_output, dim=0)
        grad_input = torch.div(gamma, B * torch.sqrt(var + eps)) * (B *
            grad_output - grad_beta - grad_gamma * normalized)
        return grad_input, grad_gamma, grad_beta, None


class CustomBatchNormManualModule(nn.Module):
    """
    This nn.module implements a custom version of the batch norm operation for MLPs.
    In self.forward the functional version CustomBatchNormManualFunction.forward is called.
    The automatic differentiation of PyTorch calls the backward method of this function in the backward pass.
    """

    def __init__(self, n_neurons, eps=1e-05):
        """
        Initializes CustomBatchNormManualModule object.

        Args:
          n_neurons: int specifying the number of neurons
          eps: small float to be added to the variance for stability
        """
        super(CustomBatchNormManualModule, self).__init__()
        self.n_neurons = n_neurons
        self.eps = eps
        self.beta = nn.Parameter(torch.zeros(self.n_neurons))
        self.gamma = nn.Parameter(torch.ones(self.n_neurons))

    def forward(self, input):
        """
        Compute the batch normalization via CustomBatchNormManualFunction

        Args:
          input: input tensor of shape (n_batch, n_neurons)
        Returns:
          out: batch-normalized tensor
        """
        assert input.shape[1] == self.n_neurons
        batch_norm_custom = CustomBatchNormManualFunction()
        out = batch_norm_custom.apply(input, self.gamma, self.beta, self.eps)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_neurons': 4}]
