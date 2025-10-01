import torch
import torch.nn as nn


class Gaussian_Kernel_Function(nn.Module):

    def __init__(self, std):
        super(Gaussian_Kernel_Function, self).__init__()
        self.sigma = std ** 2

    def forward(self, fa, fb):
        asize = fa.size()
        bsize = fb.size()
        fa1 = fa.view(-1, 1, asize[1])
        fa2 = fa.view(1, -1, asize[1])
        fb1 = fb.view(-1, 1, bsize[1])
        fb2 = fb.view(1, -1, bsize[1])
        aa = fa1 - fa2
        vaa = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(aa, 2,
            dim=2), 2), self.sigma)))
        bb = fb1 - fb2
        vbb = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(bb, 2,
            dim=2), 2), self.sigma)))
        ab = fa1 - fb2
        vab = torch.mean(torch.exp(torch.div(-torch.pow(torch.norm(ab, 2,
            dim=2), 2), self.sigma)))
        loss = vaa + vbb - 2.0 * vab
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'std': 4}]
