from torch.nn import Module
import torch
from torch import nn
import torch.utils.data
import torch.nn.functional
import torch.autograd


class DPFP(Module):
    """
    ## Deterministic Parameter Free Project (DPFP)

    This is the new projection function $\\color{lightgreen}{\\phi}$ introduced in the paper.
    DPFP projects $k$ of dimensionality $d_{key}$ to dimensionality $d_{dot} = 2 d_{key} 
u$,
    where $
u \\in \\{1, 2, ..., 2 d_{key} - 1 \\}$ is a hyper-parameter.

    $$\\color{lightgreen}{\\phi_{2 d_{key} (i - 1)  + j}(k)}
     = 	ext{ReLU}\\Big(ig[k, -kig]\\Big)_{j}
                        	ext{ReLU}\\Big(ig[k, -kig]\\Big)_{i + j}$$

    where $ig[k, -kig]$ is the concatenation of $k$ and $-k$ to give a vector of
    size $2 d_{key}$, $i \\in \\{1, 2, ..., 
u \\}$, and $j \\in \\{1, 2, ..., 2 d_{key}\\}$.
    $x_i$ is the $i$-th element of vector $x$ and is rolled around if
    $i$ is larger than the number of elements in $x$.

    Basically, it creates a new vector by multiplying elements of $[k, -k]$ shifted by $i$.

    This produces projections that are sparse (only a few elements of $phi$ are non-zero) and
    orthogonal ($\\color{lightgreen}{\\phi(k^{(i)})} \\cdot \\color{lightgreen}{\\phi(k^{(j)})}
     pprox 0$ for most $i, j$
    unless $k^{(i)}$ and $k^{(j)}$ are very similar.

    ### Normalization

    Paper introduces a simple normalization for $\\color{lightgreen}{\\phi}$,

    $$\\color{lightgreen}{\\phi '(k)} =
     rac{\\color{lightgreen}{\\phi(k)}}{\\sum^{d_{dot}}_{j=1} \\color{lightgreen}{\\phi(k)_j}}$$

    *Check the paper for derivation.*
    """

    def __init__(self, nu: 'int'=1, eps: 'float'=1e-06):
        """
        * `nu` is the hyper-parameter $
u$.
        * `eps` is the small value used to make sure there is no division-by-zero when normalizing.
        """
        super().__init__()
        self.nu = nu
        self.relu = nn.ReLU()
        self.eps = eps

    def forward(self, k: 'torch.Tensor'):
        k = self.dpfp(k)
        return k / (torch.sum(k, dim=-1, keepdim=True) + self.eps)

    def dpfp(self, k: 'torch.Tensor'):
        """
        $$\\color{lightgreen}{\\phi(k)}$$
        """
        x = self.relu(torch.cat([k, -k], dim=-1))
        x_rolled = [x.roll(shifts=i, dims=-1) for i in range(1, self.nu + 1)]
        x_rolled = torch.cat(x_rolled, dim=-1)
        x_repeat = torch.cat([x] * self.nu, dim=-1)
        return x_repeat * x_rolled


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
