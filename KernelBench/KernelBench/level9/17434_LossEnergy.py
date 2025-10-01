import torch
from torch import nn


class WaveFunctionLoss(nn.Module):
    """Base class for all wave function loss functions.

    Any such loss must be derived from the local energy and wave function
    values, :math:`L(\\{E_\\text{loc}[\\psi],\\ln|\\psi|,w\\})`, using also
    importance-sampling weights *w*.

    Shape:
        - Input1, :math:`E_\\text{loc}[\\psi](\\mathbf r)`: :math:`(*)`
        - Input2, :math:`\\ln|\\psi(\\mathbf r)|`: :math:`(*)`
        - Input3, :math:`w(\\mathbf r)`: :math:`(*)`
        - Output, *L*: :math:`()`
    """
    pass


class LossEnergy(WaveFunctionLoss):
    """Total energy loss function.

    .. math::
        L:=2\\mathbb E\\big[(E_\\text{loc}-\\mathbb E[E_\\text{loc}])\\ln|\\psi|\\big]

    Taking a derivative of only the logarithm, the resulting gradient is equivalent,
    thanks to the Hermitian property of the Hamiltonian, to the gradient of the
    plain total energy loss function, :math:`\\mathbb E[E_\\text{loc}]`.
    """

    def forward(self, Es_loc, log_psis, ws):
        assert Es_loc.grad_fn is None
        self.weights = 2 * (Es_loc - (ws * Es_loc).mean()) / len(Es_loc)
        return (self.weights * ws * log_psis).sum()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
