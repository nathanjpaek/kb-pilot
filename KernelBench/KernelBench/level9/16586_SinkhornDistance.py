import torch
import torch.utils.data


class SinkhornDistance(torch.nn.Module):
    """
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\\in\\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\\in\\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=0.001, max_iter=100, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.ones_like(mu)
        v = torch.ones_like(nu)
        for i in range(self.max_iter):
            v = self.eps * (torch.log(nu + 1e-08) - torch.logsumexp(self.M(
                C, u, v).transpose(-2, -1), dim=-1)) + v
            u = self.eps * (torch.log(mu + 1e-08) - torch.logsumexp(self.M(
                C, u, v), dim=-1)) + u
        U, V = u, v
        pi = torch.exp(self.M(C, U, V)).detach()
        cost = torch.sum(pi * C, dim=(-2, -1))
        return cost, pi

    def M(self, C, u, v):
        """
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / epsilon$"
        """
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
