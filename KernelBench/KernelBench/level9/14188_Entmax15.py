from torch.autograd import Function
import torch
from torch import nn


def _make_ix_like(X, dim):
    d = X.size(dim)
    rho = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    view = [1] * X.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _roll_last(X, dim):
    if dim == -1:
        return X
    elif dim < 0:
        dim = X.dim() - dim
    perm = [i for i in range(X.dim()) if i != dim] + [dim]
    return X.permute(perm)


def _entmax_threshold_and_support(X, dim=-1, k=None):
    """Core computation for 1.5-entmax: optimal threshold and support size.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor to compute thresholds over.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    tau : torch.Tensor like `X`, with all but the `dim` dimension intact
        the threshold value for each vector
    support_size : torch LongTensor, shape like `tau`
        the number of nonzeros in each vector.
    """
    if k is None or k >= X.shape[dim]:
        Xsrt, _ = torch.sort(X, dim=dim, descending=True)
    else:
        Xsrt, _ = torch.topk(X, k=k, dim=dim)
    rho = _make_ix_like(Xsrt, dim)
    mean = Xsrt.cumsum(dim) / rho
    mean_sq = (Xsrt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho
    delta_nz = torch.clamp(delta, 0)
    tau = mean - torch.sqrt(delta_nz)
    support_size = (tau <= Xsrt).sum(dim).unsqueeze(dim)
    tau_star = tau.gather(dim, support_size - 1)
    if k is not None and k < X.shape[dim]:
        unsolved = (support_size == k).squeeze(dim)
        if torch.any(unsolved):
            X_ = _roll_last(X, dim)[unsolved]
            tau_, ss_ = _entmax_threshold_and_support(X_, dim=-1, k=2 * k)
            _roll_last(tau_star, dim)[unsolved] = tau_
            _roll_last(support_size, dim)[unsolved] = ss_
    return tau_star, support_size


def entmax15(X, dim=-1, k=None):
    """1.5-entmax: normalizing sparse transform (a la softmax).
    Solves the optimization problem:
        max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
    where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
    Parameters
    ----------
    X : torch.Tensor
        The input tensor.
    dim : int
        The dimension along which to apply 1.5-entmax.
    k : int or None
        number of largest elements to partial-sort over. For optimal
        performance, should be slightly bigger than the expected number of
        nonzeros in the solution. If the solution is more than k-sparse,
        this function is recursively called with a 2*k schedule.
        If `None`, full sorting is performed from the beginning.
    Returns
    -------
    P : torch tensor, same shape as X
        The projection result, such that P.sum(dim=dim) == 1 elementwise.
    """
    return Entmax15Function.apply(X, dim, k)


class Entmax15Function(Function):

    @classmethod
    def forward(cls, ctx, X, dim=0, k=None):
        ctx.dim = dim
        max_val, _ = X.max(dim=dim, keepdim=True)
        X = X - max_val
        X = X / 2
        tau_star, _ = _entmax_threshold_and_support(X, dim=dim, k=k)
        Y = torch.clamp(X - tau_star, min=0) ** 2
        ctx.save_for_backward(Y)
        return Y

    @classmethod
    def backward(cls, ctx, dY):
        Y, = ctx.saved_tensors
        gppr = Y.sqrt()
        dX = dY * gppr
        q = dX.sum(ctx.dim) / gppr.sum(ctx.dim)
        q = q.unsqueeze(ctx.dim)
        dX -= q * gppr
        return dX, None, None


class Entmax15(nn.Module):

    def __init__(self, dim=-1, k=None):
        """1.5-entmax: normalizing sparse transform (a la softmax).
        Solves the optimization problem:
            max_p <x, p> - H_1.5(p)    s.t.    p >= 0, sum(p) == 1.
        where H_1.5(p) is the Tsallis alpha-entropy with alpha=1.5.
        Parameters
        ----------
        dim : int
            The dimension along which to apply 1.5-entmax.
        k : int or None
            number of largest elements to partial-sort over. For optimal
            performance, should be slightly bigger than the expected number of
            nonzeros in the solution. If the solution is more than k-sparse,
            this function is recursively called with a 2*k schedule.
            If `None`, full sorting is performed from the beginning.
        """
        self.dim = dim
        self.k = k
        super(Entmax15, self).__init__()

    def forward(self, X):
        return entmax15(X, dim=self.dim, k=self.k)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
