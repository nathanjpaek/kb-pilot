import torch
from torch.nn.parameter import Parameter


class AdaptiveTanh(torch.nn.Module):
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
        super(AdaptiveTanh, self).__init__()
        if alpha is None:
            self.alpha = Parameter(torch.tensor(1.0))
        else:
            self.alpha = Parameter(torch.tensor(alpha))
        self.alpha.requiresGrad = True
        self.scale = Parameter(torch.tensor(1.0))
        self.scale.requiresGrad = True
        self.translate = Parameter(torch.tensor(0.0))
        self.translate.requiresGrad = True

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        """
        x += self.translate
        return self.scale * (torch.exp(self.alpha * x) - torch.exp(-self.
            alpha * x)) / (torch.exp(self.alpha * x) + torch.exp(-self.
            alpha * x))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
