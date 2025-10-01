from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class CrossEntropyBayesRisk(Module):
    """
    <a id="CrossEntropyBayesRisk"></a>

    ## Bayes Risk with Cross Entropy Loss

    Bayes risk is the overall maximum cost of making incorrect estimates.
    It takes a cost function that gives the cost of making an incorrect estimate
    and sums it over all possible outcomes based on probability distribution.

    Here the cost function is cross-entropy loss, for one-hot coded $\\mathbf{y}$
    $$\\sum_{k=1}^K -y_k \\log p_k$$

    We integrate this cost over all $\\mathbf{p}$

    egin{align}
    \\mathcal{L}(\\Theta)
    &= -\\log \\Bigg(
     \\int
      \\Big[ \\sum_{k=1}^K -y_k \\log p_k \\Big]
      rac{1}{B(	extcolor{orange}{\\mathbf{lpha}})}
      \\prod_{k=1}^K p_k^{	extcolor{orange}{lpha_k} - 1}
     d\\mathbf{p}
     \\Bigg ) \\
    &= \\sum_{k=1}^K y_k igg( \\psi(S) - \\psi( 	extcolor{orange}{lpha_k} ) igg)
    \\end{align}

    where $\\psi(\\cdot)$ is the $digamma$ function.
    """

    def forward(self, evidence: 'torch.Tensor', target: 'torch.Tensor'):
        """
        * `evidence` is $\\mathbf{e} \\ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        loss = (target * (torch.digamma(strength)[:, None] - torch.digamma(
            alpha))).sum(dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
