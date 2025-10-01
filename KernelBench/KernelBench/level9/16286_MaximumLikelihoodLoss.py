from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class MaximumLikelihoodLoss(Module):
    """
    <a id="MaximumLikelihoodLoss"></a>

    ## Type II Maximum Likelihood Loss

    The distribution $D(\\mathbf{p} ert 	extcolor{orange}{\\mathbf{lpha}})$ is a prior on the likelihood
    $Multi(\\mathbf{y} ert p)$,
     and the negative log marginal likelihood is calculated by integrating over class probabilities
     $\\mathbf{p}$.

    If target probabilities (one-hot targets) are $y_k$ for a given sample the loss is,

    egin{align}
    \\mathcal{L}(\\Theta)
    &= -\\log \\Bigg(
     \\int
      \\prod_{k=1}^K p_k^{y_k}
      rac{1}{B(	extcolor{orange}{\\mathbf{lpha}})}
      \\prod_{k=1}^K p_k^{	extcolor{orange}{lpha_k} - 1}
     d\\mathbf{p}
     \\Bigg ) \\
    &= \\sum_{k=1}^K y_k igg( \\log S - \\log 	extcolor{orange}{lpha_k} igg)
    \\end{align}
    """

    def forward(self, evidence: 'torch.Tensor', target: 'torch.Tensor'):
        """
        * `evidence` is $\\mathbf{e} \\ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        loss = (target * (strength.log()[:, None] - alpha.log())).sum(dim=-1)
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
