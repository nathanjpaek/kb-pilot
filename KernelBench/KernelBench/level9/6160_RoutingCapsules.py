import torch
import torch.nn as nn
import torch.nn.functional as F


def squash(x, dim=-1, epsilon=1e-08):
    norm = (x ** 2).sum(dim=dim, keepdim=True)
    x = norm / (norm + 1) * x / (torch.sqrt(norm) + epsilon)
    return x


class RoutingCapsules(nn.Module):
    """
    input
        capsules_num: new feature, num to duplicate
        capsules_dim: new feature, each capsules' dim
        in_capsules_num: last layer's capsules_num
        in_capsules_dim: last layer's capsules_dim
    """

    def __init__(self, capsules_num, capsules_dim, in_capsules_num,
        in_capsules_dim, num_iterations=3):
        super(RoutingCapsules, self).__init__()
        self.capsules_num = capsules_num
        self.capsules_dim = capsules_dim
        self.in_capsules_num = in_capsules_num
        self.in_capsules_dim = in_capsules_dim
        self.num_iterations = num_iterations
        self.W = nn.Parameter(torch.randn(1, capsules_num, in_capsules_num,
            in_capsules_dim, capsules_dim))

    def forward(self, x):
        x.shape[0]
        x = x.unsqueeze(1).unsqueeze(3)
        u_hat = x @ self.W
        b = torch.zeros_like(u_hat)
        for i in range(self.num_iterations - 1):
            """
            Softmax is applied on all of the input capsules, to calculate probs.
            """
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=2, keepdim=True)
            v = squash(s)
            uv = u_hat * v
            b = b + uv
        c = F.softmax(b, dim=2)
        s = (c * u_hat).sum(dim=2, keepdim=True)
        v = squash(s)
        v = v.squeeze()
        return v


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'capsules_num': 4, 'capsules_dim': 4, 'in_capsules_num': 4,
        'in_capsules_dim': 4}]
