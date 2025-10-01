import torch
import numpy as np
import torch.nn as nn


def sim_to_dist(scores):
    return 1 - torch.sqrt(2.001 - 2 * scores)


class APLoss(nn.Module):
    """ Differentiable AP loss, through quantization. From the paper:

        Learning with Average Precision: Training Image Retrieval with a Listwise Loss
        Jerome Revaud, Jon Almazan, Rafael Sampaio de Rezende, Cesar de Souza
        https://arxiv.org/abs/1906.07589

        Input: (N, M)   values in [min, max]
        label: (N, M)   values in {0, 1}

        Returns: 1 - mAP (mean AP for each n in {1..N})
                 Note: typically, this is what you wanna minimize
    """

    def __init__(self, nq=25, min=0, max=1):
        nn.Module.__init__(self)
        assert isinstance(nq, int) and 2 <= nq <= 100
        self.nq = nq
        self.min = min
        self.max = max
        gap = max - min
        assert gap > 0
        self.quantizer = q = nn.Conv1d(1, 2 * nq, kernel_size=1, bias=True)
        q.weight = nn.Parameter(q.weight.detach(), requires_grad=False)
        q.bias = nn.Parameter(q.bias.detach(), requires_grad=False)
        a = (nq - 1) / gap
        q.weight[:nq] = -a
        q.bias[:nq] = torch.from_numpy(a * min + np.arange(nq, 0, -1))
        q.weight[nq:] = a
        q.bias[nq:] = torch.from_numpy(np.arange(2 - nq, 2, 1) - a * min)
        q.weight[0] = q.weight[-1] = 0
        q.bias[0] = q.bias[-1] = 1

    def forward(self, x, label, qw=None, ret='1-mAP'):
        assert x.shape == label.shape
        N, M = x.shape
        q = self.quantizer(x.unsqueeze(1))
        q = torch.min(q[:, :self.nq], q[:, self.nq:]).clamp(min=0)
        nbs = q.sum(dim=-1)
        rec = (q * label.view(N, 1, M).float()).sum(dim=-1)
        prec = rec.cumsum(dim=-1) / (1e-16 + nbs.cumsum(dim=-1))
        rec /= rec.sum(dim=-1).unsqueeze(1)
        ap = (prec * rec).sum(dim=-1)
        if ret == '1-mAP':
            if qw is not None:
                ap *= qw
            return 1 - ap.mean()
        elif ret == 'AP':
            assert qw is None
            return ap
        else:
            raise ValueError('Bad return type for APLoss(): %s' % str(ret))

    def measures(self, x, gt, loss=None):
        if loss is None:
            loss = self.forward(x, gt)
        return {'loss_ap': float(loss)}


class APLoss_dist(APLoss):

    def forward(self, x, label, **kw):
        d = sim_to_dist(x)
        return APLoss.forward(self, d, label, **kw)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
