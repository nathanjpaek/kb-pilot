import torch
from torch import nn
from sklearn.preprocessing import scale as scale


def Covariance(m, bias=False, rowvar=True, inplace=False):
    """ Estimate a covariance matrix given data(tensor).
    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: numpy array - A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: bool - If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    """
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    fact = 1.0 / (m.size(1) - 1) if not bias else 1.0 / m.size(1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()
    return fact * m.matmul(mt).squeeze()


class CMDS_Loss(nn.Module):
    """Equation(1) in Self-calibrating Neural Networks for Dimensionality Reduction

    Attributes:
        X: tensor - original datas.
        Y: tensor - encoded datas.
    Returns:
        cmds: float - The cmds loss.
    """

    def __init__(self):
        super(CMDS_Loss, self).__init__()

    def forward(self, y, x):
        XTX = Covariance(x.T, bias=True)
        YTY = Covariance(y.T, bias=True)
        cmds = torch.norm(XTX - YTY) ** 2
        return cmds


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
