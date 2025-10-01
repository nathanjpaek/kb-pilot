import torch
import torch.nn as nn


class CRFLoss(nn.Module):

    def __init__(self, L, init):
        super(CRFLoss, self).__init__()
        self.start = nn.Parameter(torch.Tensor(L).uniform_(-init, init))
        self.T = nn.Parameter(torch.Tensor(L, L).uniform_(-init, init))
        self.end = nn.Parameter(torch.Tensor(L).uniform_(-init, init))

    def forward(self, scores, targets):
        normalizers = self.compute_normalizers(scores)
        target_scores = self.score_targets(scores, targets)
        loss = (normalizers - target_scores).mean()
        return loss

    def decode(self, scores):
        _B, T, _L = scores.size()
        prev = self.start.unsqueeze(0) + scores[:, 0]
        back = []
        for i in range(1, T):
            cur = prev.unsqueeze(2) + scores.transpose(0, 1)[i].unsqueeze(1
                ) + self.T.transpose(0, 1)
            prev, indices = cur.max(dim=1)
            back.append(indices)
        prev += self.end
        max_scores, indices = prev.max(dim=1)
        tape = [indices]
        back = list(reversed(back))
        for i in range(T - 1):
            indices = torch.gather(back[i], 1, indices.unsqueeze(1)).squeeze(1)
            tape.append(indices)
        return max_scores, torch.stack(tape[::-1], dim=1)

    def compute_normalizers(self, scores):
        _B, T, _L = scores.size()
        prev = self.start + scores.transpose(0, 1)[0]
        for i in range(1, T):
            cur = prev.unsqueeze(2) + scores.transpose(0, 1)[i].unsqueeze(1
                ) + self.T.transpose(0, 1)
            prev = torch.logsumexp(cur, dim=1).clone()
        prev += self.end
        normalizers = torch.logsumexp(prev, 1)
        return normalizers

    def score_targets(self, scores, targets):
        _B, T, _L = scores.size()
        emits = scores.gather(2, targets.unsqueeze(2)).squeeze(2).sum(1)
        trans = torch.stack([self.start.gather(0, targets[:, 0])] + [self.T
            [targets[:, i], targets[:, i - 1]] for i in range(1, T)] + [
            self.end.gather(0, targets[:, -1])]).sum(0)
        return emits + trans


def get_inputs():
    return [torch.ones([4, 4, 4], dtype=torch.int64), torch.ones([4, 4],
        dtype=torch.int64)]


def get_init_inputs():
    return [[], {'L': 4, 'init': 4}]
