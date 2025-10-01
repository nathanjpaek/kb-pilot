import torch
import torch.nn as nn
from torch import sin
from torch import pow
from torch.nn import Parameter
from torch.distributions.exponential import Exponential


class Snake(nn.Module):
    """         
    Implementation of the serpentine-like sine-based periodic activation function
    
    .. math::
         Snake_a := x + rac{1}{a} sin^2(ax) = x - rac{1}{2a}cos{2ax} + rac{1}{2a}

    Shape:
        - Input: (N, *) where * means, any number of additional
          dimensions
        - Output: (N, *), same shape as the input
        
    Parameters:
        - a - trainable parameter
    
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
        
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(self, in_features, a=None, trainable=True):
        """
        Initialization.
        Args:
            in_features: shape of the input
            a: trainable parameter
            trainable: sets `a` as a trainable parameter
            
            `a` is initialized to 1 by default, higher values = higher-frequency, 
            5-50 is a good starting point if you already think your data is periodic, 
            consider starting lower e.g. 0.5 if you think not, but don't worry, 
            `a` will be trained along with the rest of your model. 
        """
        super(Snake, self).__init__()
        self.in_features = in_features if isinstance(in_features, list) else [
            in_features]
        if a is not None:
            self.a = Parameter(torch.ones(self.in_features) * a)
        else:
            m = Exponential(torch.tensor([0.1]))
            self.a = Parameter(m.rsample(self.in_features).squeeze())
        self.a.requiresGrad = trainable

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a* sin^2 (xa)
        """
        return x + 1.0 / self.a * pow(sin(x * self.a), 2)


class SnakeHyperSolver(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.a1 = Snake(hidden_dim)
        self.a2 = Snake(hidden_dim)

    def forward(self, x):
        x = self.a1(self.fc1(x))
        x = self.a2(self.fc2(x))
        x = self.fc3(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_dim': 4, 'out_dim': 4}]
