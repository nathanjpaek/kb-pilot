from torch.nn import Module
import torch
import torch.utils.data
import torch.nn.functional
import torch.autograd


class KLDivergenceLoss(Module):
    """
    <a id="KLDivergenceLoss"></a>

    ## KL Divergence Regularization Loss

    This tries to shrink the total evidence to zero if the sample cannot be correctly classified.

    First we calculate $	ilde{lpha}_k = y_k + (1 - y_k) 	extcolor{orange}{lpha_k}$ the
    Dirichlet parameters after remove the correct evidence.

    egin{align}
    &KL \\Big[ D(\\mathbf{p} ert \\mathbf{	ilde{lpha}}) \\Big \\Vert
    D(\\mathbf{p} ert <1, \\dots, 1>\\Big] \\
    &= \\log \\Bigg( rac{\\Gamma \\Big( \\sum_{k=1}^K 	ilde{lpha}_k \\Big)}
    {\\Gamma(K) \\prod_{k=1}^K \\Gamma(	ilde{lpha}_k)} \\Bigg)
    + \\sum_{k=1}^K (	ilde{lpha}_k - 1)
    \\Big[ \\psi(	ilde{lpha}_k) - \\psi(	ilde{S}) \\Big]
    \\end{align}

    where $\\Gamma(\\cdot)$ is the gamma function,
    $\\psi(\\cdot)$ is the $digamma$ function and
    $	ilde{S} = \\sum_{k=1}^K 	ilde{lpha}_k$
    """

    def forward(self, evidence: 'torch.Tensor', target: 'torch.Tensor'):
        """
        * `evidence` is $\\mathbf{e} \\ge 0$ with shape `[batch_size, n_classes]`
        * `target` is $\\mathbf{y}$ with shape `[batch_size, n_classes]`
        """
        alpha = evidence + 1.0
        n_classes = evidence.shape[-1]
        alpha_tilde = target + (1 - target) * alpha
        strength_tilde = alpha_tilde.sum(dim=-1)
        first = torch.lgamma(alpha_tilde.sum(dim=-1)) - torch.lgamma(
            alpha_tilde.new_tensor(float(n_classes))) - torch.lgamma(
            alpha_tilde).sum(dim=-1)
        second = ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.
            digamma(strength_tilde)[:, None])).sum(dim=-1)
        loss = first + second
        return loss.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
