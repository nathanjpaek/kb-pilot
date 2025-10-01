from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class Squash(Module):
    '\n    ## Squash\n\n    This is **squashing** function from paper, given by equation $(1)$.\n\n    $$\\mathbf{v}_j = \x0crac{{\\lVert \\mathbf{s}_j \rVert}^2}{1 + {\\lVert \\mathbf{s}_j \rVert}^2}\n     \x0crac{\\mathbf{s}_j}{\\lVert \\mathbf{s}_j \rVert}$$\n\n    $\x0crac{\\mathbf{s}_j}{\\lVert \\mathbf{s}_j \rVert}$\n    normalizes the length of all the capsules, whilst\n    $\x0crac{{\\lVert \\mathbf{s}_j \rVert}^2}{1 + {\\lVert \\mathbf{s}_j \rVert}^2}$\n    shrinks the capsules that have a length smaller than one .\n    '

    def __init__(self, epsilon=1e-08):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, s: 'torch.Tensor'):
        """
        The shape of `s` is `[batch_size, n_capsules, n_features]`
        """
        s2 = (s ** 2).sum(dim=-1, keepdims=True)
        return s2 / (1 + s2) * (s / torch.sqrt(s2 + self.epsilon))


class Router(Module):
    """
    ## Routing Algorithm

    This is the routing mechanism described in the paper.
    You can use multiple routing layers in your models.

    This combines calculating $\\mathbf{s}_j$ for this layer and
    the routing algorithm described in *Procedure 1*.
    """

    def __init__(self, in_caps: 'int', out_caps: 'int', in_d: 'int', out_d:
        'int', iterations: 'int'):
        """
        `in_caps` is the number of capsules, and `in_d` is the number of features per capsule from the layer below.
        `out_caps` and `out_d` are the same for this layer.

        `iterations` is the number of routing iterations, symbolized by $r$ in the paper.
        """
        super().__init__()
        self.in_caps = in_caps
        self.out_caps = out_caps
        self.iterations = iterations
        self.softmax = nn.Softmax(dim=1)
        self.squash = Squash()
        self.weight = nn.Parameter(torch.randn(in_caps, out_caps, in_d,
            out_d), requires_grad=True)

    def forward(self, u: 'torch.Tensor'):
        """
        The shape of `u` is `[batch_size, n_capsules, n_features]`.
        These are the capsules from the lower layer.
        """
        u_hat = torch.einsum('ijnm,bin->bijm', self.weight, u)
        b = u.new_zeros(u.shape[0], self.in_caps, self.out_caps)
        v = None
        for i in range(self.iterations):
            c = self.softmax(b)
            s = torch.einsum('bij,bijm->bjm', c, u_hat)
            v = self.squash(s)
            a = torch.einsum('bjm,bijm->bij', v, u_hat)
            b = b + a
        return v


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'in_caps': 4, 'out_caps': 4, 'in_d': 4, 'out_d': 4,
        'iterations': 4}]
