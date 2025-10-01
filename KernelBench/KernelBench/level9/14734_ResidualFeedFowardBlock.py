import torch


class ResidualFeedFowardBlock(torch.nn.Module):
    """Block of two feed-forward layer with a reisdual connection:
      
            f(W1^T x + b1)         f(W2^T h1 + b2 )         h2 + x 
        x ------------------> h1 --------------------> h2 ----------> y
        |                                              ^
        |               Residual connection            | 
        +----------------------------------------------+
        
    """

    def __init__(self, dim_in, width, activation_fn=torch.nn.Tanh):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_in, width)
        self.layer2 = torch.nn.Linear(width, dim_in)
        self.activation_fn = activation_fn()

    def forward(self, x):
        h1 = self.activation_fn(self.layer1(x))
        h2 = self.activation_fn(self.layer2(h1))
        return h2 + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'width': 4}]
