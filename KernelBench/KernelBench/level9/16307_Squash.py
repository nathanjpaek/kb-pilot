from torch.nn import Module
import torch
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


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
