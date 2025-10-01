import torch
from torch.nn.parameter import Parameter


class AdaptiveSin(torch.nn.Module):
    """
    Implementation of soft exponential activation.
    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - See related paper:
        https://arxiv.org/pdf/1602.01321.pdf
    Examples:
        >>> a1 = soft_exponential(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, alpha=None):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - aplha: trainable parameter
            aplha is initialized with zero value by default
        """
        super(AdaptiveSin, self).__init__()
        self.alpha = Parameter(torch.normal(torch.tensor(1.0), torch.tensor
            (0.1)))
        self.alpha.requiresGrad = True
        self.scale = Parameter(torch.normal(torch.tensor(1.0), torch.tensor
            (0.1)))
        self.scale.requiresGrad = True
        self.translate = Parameter(torch.normal(torch.tensor(0.0), torch.
            tensor(0.1)))
        self.translate.requiresGrad = True

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        return self.scale * torch.sin(self.alpha * x + self.translate)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
