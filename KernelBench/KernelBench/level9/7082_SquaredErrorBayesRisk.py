from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class SquaredErrorBayesRisk(Module):
    """
    <a id="SquaredErrorBayesRisk"></a>

    ## Bayes Risk with Squared Error Loss

    Here the cost function is squared error,
    $$\\sum_{k=1}^K (y_k - p_k)^2 = \\Vert \\mathbf{y} - \\mathbf{p} \\Vert_2^2$$

    We integrate this cost over all $\\mathbf{p}$

    egin{align}
    \\mathcal{L}(\\Theta)
    &= -\\log \\Bigg(
     \\int
      \\Big[ \\sum_{k=1}^K (y_k - p_k)^2 \\Big]
      rac{1}{B(	extcolor{orange}{\\mathbf{lpha}})}
      \\prod_{k=1}^K p_k^{	extcolor{orange}{lpha_k} - 1}
     d\\mathbf{p}
     \\Bigg ) \\
    &= \\sum_{k=1}^K \\mathbb{E} \\Big[ y_k^2 -2 y_k p_k + p_k^2 \\Big] \\
    &= \\sum_{k=1}^K \\Big( y_k^2 -2 y_k \\mathbb{E}[p_k] + \\mathbb{E}[p_k^2] \\Big)
    \\end{align}

    Where $$\\mathbb{E}[p_k] = \\hat{p}_k = rac{	extcolor{orange}{lpha_k}}{S}$$
    is the expected probability when sampled from the Dirichlet distribution
    and $$\\mathbb{E}[p_k^2] = \\mathbb{E}[p_k]^2 + 	ext{Var}(p_k)$$
     where
    $$	ext{Var}(p_k) = rac{	extcolor{orange}{lpha_k}(S - 	extcolor{orange}{lpha_k})}{S^2 (S + 1)}
    = rac{\\hat{p}_k(1 - \\hat{p}_k)}{S + 1}$$
     is the variance.

    This gives,

    egin{align}
    \\mathcal{L}(\\Theta)
    &= \\sum_{k=1}^K \\Big( y_k^2 -2 y_k \\mathbb{E}[p_k] + \\mathbb{E}[p_k^2] \\Big) \\
    &= \\sum_{k=1}^K \\Big( y_k^2 -2 y_k \\mathbb{E}[p_k] +  \\mathbb{E}[p_k]^2 + 	ext{Var}(p_k) \\Big) \\
    &= \\sum_{k=1}^K \\Big( ig( y_k -\\mathbb{E}[p_k] ig)^2 + 	ext{Var}(p_k) \\Big) \\
    &= \\sum_{k=1}^K \\Big( ( y_k -\\hat{p}_k)^2 + rac{\\hat{p}_k(1 - \\hat{p}_k)}{S + 1} \\Big)
    \\end{align}

    This first part of the equation $ig(y_k -\\mathbb{E}[p_k]ig)^2$ is the error term and
    the second part is the variance.
    """

    def forward(self, evidence: 'torch.Tensor', target: 'torch.Tensor'):
        """
        * `evidence` is $\\mathbf{e} \\ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]
        err = (target - p) ** 2
        var = p * (1 - p) / (strength[:, None] + 1)
        loss = (err + var).sum(dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
