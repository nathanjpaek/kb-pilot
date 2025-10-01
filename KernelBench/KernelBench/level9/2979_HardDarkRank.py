import torch
import torch.nn as nn


def pdist(e, squared=False, eps=1e-12):
    e_square = e.pow(2).sum(dim=1)
    prod = e @ e.t()
    res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min
        =eps)
    if not squared:
        res = res.sqrt()
    res = res.clone()
    res[range(len(e)), range(len(e))] = 0
    return res


class HardDarkRank(nn.Module):

    def __init__(self, alpha=3, beta=3, permute_len=4):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.permute_len = permute_len

    def forward(self, student, teacher):
        score_teacher = -1 * self.alpha * pdist(teacher, squared=False).pow(
            self.beta)
        score_student = -1 * self.alpha * pdist(student, squared=False).pow(
            self.beta)
        permute_idx = score_teacher.sort(dim=1, descending=True)[1][:, 1:
            self.permute_len + 1]
        ordered_student = torch.gather(score_student, 1, permute_idx)
        log_prob = (ordered_student - torch.stack([torch.logsumexp(
            ordered_student[:, i:], dim=1) for i in range(permute_idx.size(
            1))], dim=1)).sum(dim=1)
        loss = (-1 * log_prob).mean()
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
