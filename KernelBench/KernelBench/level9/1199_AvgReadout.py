import torch
import torch.nn as nn


class AvgReadout(nn.Module):
    """
    Considering the efficiency of the method, we simply employ average pooling, computing the average of the set of embedding matrices

    .. math::
      \\begin{equation}
        \\mathbf{H}=\\mathcal{Q}\\left(\\left\\{\\mathbf{H}^{(r)} \\mid r \\in \\mathcal{R}\\right\\}\\right)=\\frac{1}{|\\mathcal{R}|} \\sum_{r \\in \\mathcal{R}} \\mathbf{H}^{(r)}
      \\end{equation}
    """

    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
