import torch
import torch.nn as nn
import torch.utils.data


class QuadraticForm(torch.autograd.Function):
    """
    This is a custom function that, given two parameters mew and sigma, implements quadratic form. 
    This function takes a representation of a color in vector space and returns a unnormalized score attributed to that color swab.
    """

    @staticmethod
    def forward(ctx, mew, co_var, color):
        """        
        mew : FloatTensor
            m x k matrix, where m is the number of examples, where k is the length of the representation of the color
        co_var: FloatTensor
            m x k x k matrix, sigma in the quadratic form. 
        color: FloatTensor
            m x p x k matrix, where each example has a p vectors of a single color representations of length k
        ------
        outputs:
            m x p matrix of scores.

        """
        ctx.save_for_backward(mew, co_var, color)
        shifted_color = color - mew.unsqueeze(1)
        vec_mat_mult = -torch.matmul(shifted_color.unsqueeze(2), co_var.
            unsqueeze(1)).squeeze(1)
        output = (vec_mat_mult.squeeze(2) * shifted_color).sum(2)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        The derivative of the quadratic form.
        
        input : tuple of FloatTensors
            mew : FloatTensor
                m x k matrix, where m is the number of examples, where k is the length of the representation of the color
            co_var: FloatTensor
                m x k x k matrix, sigma in the quadratic form
            color: FloatTensor
                m x k matrix, where each example has a vector of a single color representation of length k
        output : FloatTensor
            The float tensor for the gradients of a quadratic function
        """
        mew, co_var, color = ctx.saved_tensors
        grad_mean = grad_co_var = grad_color = None
        shifted_color = color - mew.unsqueeze(1)
        if ctx.needs_input_grad[0]:
            grad_mean = torch.matmul(shifted_color.unsqueeze(2), (co_var +
                co_var.permute(0, 2, 1)).unsqueeze(1)).squeeze(2)
            grad_mean = grad_mean * grad_output.unsqueeze(2)
            grad_mean = grad_mean.sum(1)
        if ctx.needs_input_grad[1]:
            grad_co_var = -torch.einsum('bki,bkj->bkij', (shifted_color,
                shifted_color))
            grad_co_var = grad_co_var * grad_output.unsqueeze(2).unsqueeze(3)
            grad_co_var = grad_co_var.sum(1)
        return grad_mean, grad_co_var, grad_color


class AttentionalColorizedListenerDecoder(nn.Module):
    """
    Simple decoder model for the neural literal/pragmatic listener.
    This model takes in two statistical params, mew and sigma, and returns a vector containing the normalized scores
    of each color in the context.
    """

    def __init__(self, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = device
        self.transform_func = QuadraticForm.apply
        self.hidden_activation = nn.Softmax(dim=1)

    def forward(self, color_seqs, mew, sigma):
        """
        color_seqs : FloatTensor
            A m x k x n tensor where m is the number of examples, k is the number of colors in the context, and
            n is the size of the color dimension after transform
        """
        color_scores = self.transform_func(mew, sigma, color_seqs)
        output = self.hidden_activation(color_scores)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'device': 0}]
