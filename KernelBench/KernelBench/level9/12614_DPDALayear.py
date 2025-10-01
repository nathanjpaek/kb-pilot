import torch
from torch import nn


class DPDALayear(nn.Module):

    def __init__(self, dim):
        super(DPDALayear, self).__init__()
        self.W_p = nn.Linear(2 * dim, dim)
        self.W_q = nn.Linear(2 * dim, dim)

    def forward(self, P, Q, p_mask=None, q_mask=None):
        P_ori = P
        Q_ori = Q
        A = torch.matmul(P, Q.transpose(dim0=1, dim1=2))
        if p_mask is not None:
            p_mask = p_mask.float()
            p_mask = 1 - p_mask
            p_mask = p_mask * -10000.0
            p_mask = p_mask.unsqueeze(dim=2)
            p_mask = p_mask.expand_as(A)
            A = A + p_mask
        if q_mask is not None:
            q_mask = q_mask.float()
            q_mask = 1 - q_mask
            q_mask = q_mask * -10000.0
            q_mask = q_mask.unsqueeze(dim=1)
            q_mask = q_mask.expand_as(A)
            A = A + q_mask
        A_q = torch.softmax(A, dim=2)
        A_p = torch.softmax(A.transpose(dim0=1, dim1=2), dim=2)
        P_q = torch.matmul(A_q, Q)
        Q_p = torch.matmul(A_p, P)
        P_t = torch.cat([P_q, P], dim=2)
        Q_t = torch.cat([Q_p, Q], dim=2)
        Q = torch.matmul(A_p, P_t)
        P = torch.matmul(A_q, Q_t)
        P = P_ori + self.W_p(P)
        Q = Q_ori + self.W_q(Q)
        return P, Q


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
