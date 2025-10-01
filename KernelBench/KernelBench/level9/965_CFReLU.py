import torch
import torch.nn as nn
import torch.nn.functional as F


class CFReLU(nn.Module):
    """
     Custom FReLU
    
    cfrelu(x) = relu(x + a) + b
    see psychXRF.activation.FReLU
    
    Shape:
        -Input: (N, *)
        -Output: (N, *), same shape as the input
        
    Parameters:
        -a: trainable parameter
        -b: fixed parameter
    
    Examples:
        >>> input = torch.randn(300, 6)
        >>> act = CFReLU(input.shape[1], b = 1.0e-6)
        >>> x = act(input)
    """

    def __init__(self, in_features, b, a=None):
        """
        Initialization
        
        a is initialized with zero value by default
        """
        super(CFReLU, self).__init__()
        self.relu = F.relu
        self.in_features = in_features
        self.b = nn.Parameter(torch.tensor(b), requires_grad=False)
        if a:
            self.a = nn.Parameter(torch.tensor(a))
        else:
            self.a = nn.Parameter(torch.tensor(0.0))
        self.a.requiresGrad = True

    def forward(self, x):
        return self.relu(x + self.a) + self.b


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'b': 4}]
