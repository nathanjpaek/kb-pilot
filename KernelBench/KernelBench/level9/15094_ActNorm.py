import torch
import torch.utils.data


class ActNorm(torch.nn.Module):

    def __init__(self, nsq, data_init=True):
        super(ActNorm, self).__init__()
        self.initialized = not data_init
        self.m = torch.nn.Parameter(torch.zeros(1, nsq, 1))
        self.logs = torch.nn.Parameter(torch.zeros(1, nsq, 1))
        return

    def forward(self, h):
        if not self.initialized:
            _sbatch, nsq, _lchunk = h.size()
            flatten = h.permute(1, 0, 2).contiguous().view(nsq, -1).data
            self.m.data = -flatten.mean(1).view(1, nsq, 1)
            self.logs.data = torch.log(1 / (flatten.std(1) + 1e-07)).view(1,
                nsq, 1)
            self.initialized = True
        h = torch.exp(self.logs) * (h + self.m)
        logdet = self.logs.sum() * h.size(2)
        return h, logdet

    def reverse(self, h):
        return h * torch.exp(-self.logs) - self.m


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'nsq': 4}]
