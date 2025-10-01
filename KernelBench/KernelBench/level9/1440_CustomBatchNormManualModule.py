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

    TODO:
      Implement the forward pass of batch normalization
      Store constant non-tensor objects via ctx.constant=myconstant
      Store tensors which you need in the backward pass via ctx.save_for_backward(tensor1, tensor2, ...)
      Intermediate results can be decided to be either recomputed in the backward pass or to be stored
      for the backward pass. Do not store tensors which are unnecessary for the backward pass to save memory!
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
        elif shape[1] != gamma.shape[0]:
            raise ValueError(
                f'Expected input of shape batch_size x {gamma.shape[0]}. Instead, got input withshape of {shape}.'
                )
        mean = input.mean(0)
        var = input.var(0, unbiased=False)
        inv_var = 1 / torch.sqrt(var + eps)
        x_hat = (input - mean) * inv_var
        out = gamma * x_hat + beta
        ctx.save_for_backward(input, gamma, beta, mean, inv_var)
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
    
    TODO:
      Retrieve saved tensors and constants via ctx.saved_tensors and ctx.constant
      Compute gradients for inputs where ctx.needs_input_grad[idx] is True. Set gradients for other
      inputs to None. This should be decided dynamically.
    """
        input, gamma, _beta, mean, inv_var = ctx.saved_tensors
        input_needs_grad, gamma_needs_grad, beta_needs_grad = (ctx.
            needs_input_grad)
        batch_dim = input.shape[0]
        x_hat = (input - mean) * inv_var
        dx_hat = grad_output * gamma
        grad_gamma = (grad_output * x_hat).sum(0) if gamma_needs_grad else None
        grad_beta = grad_output.sum(0) if beta_needs_grad else None
        grad_input = 1.0 / batch_dim * inv_var * (batch_dim * dx_hat -
            dx_hat.sum(0) - x_hat * (dx_hat * x_hat).sum(0)
            ) if input_needs_grad else None
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
    
    TODO:
      Save parameters for the number of neurons and eps.
      Initialize parameters gamma and beta via nn.Parameter
    """
        super(CustomBatchNormManualModule, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_neurons))
        self.beta = nn.Parameter(torch.zeros(n_neurons))
        self.eps = eps

    def forward(self, input):
        """
    Compute the batch normalization via CustomBatchNormManualFunction
    
    Args:
      input: input tensor of shape (n_batch, n_neurons)
    Returns:
      out: batch-normalized tensor
    
    TODO:
      Check for the correctness of the shape of the input tensor.
      Instantiate a CustomBatchNormManualFunction.
      Call it via its .apply() method.
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
        batchNormFunc = CustomBatchNormManualFunction()
        out = batchNormFunc.apply(input, self.gamma, self.beta, self.eps)
        return out


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'n_neurons': 4}]
