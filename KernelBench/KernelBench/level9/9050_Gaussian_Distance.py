import torch
import torch.nn as nn


class Gaussian_Distance(nn.Module):

    def __init__(self, kern=1):
        super(Gaussian_Distance, self).__init__()
        self.kern = kern
        self.avgpool = nn.AvgPool2d(kernel_size=kern, stride=kern)

    def forward(self, mu_a, logvar_a, mu_b, logvar_b):
        mu_a = self.avgpool(mu_a)
        mu_b = self.avgpool(mu_b)
        var_a = self.avgpool(torch.exp(logvar_a)) / (self.kern * self.kern)
        var_b = self.avgpool(torch.exp(logvar_b)) / (self.kern * self.kern)
        mu_a1 = mu_a.view(mu_a.size(0), 1, -1)
        mu_a2 = mu_a.view(1, mu_a.size(0), -1)
        var_a1 = var_a.view(var_a.size(0), 1, -1)
        var_a2 = var_a.view(1, var_a.size(0), -1)
        mu_b1 = mu_b.view(mu_b.size(0), 1, -1)
        mu_b2 = mu_b.view(1, mu_b.size(0), -1)
        var_b1 = var_b.view(var_b.size(0), 1, -1)
        var_b2 = var_b.view(1, var_b.size(0), -1)
        vaa = torch.sum(torch.exp(torch.mul(torch.add(torch.div(torch.pow(
            mu_a1 - mu_a2, 2), var_a1 + var_a2), torch.log(var_a1 + var_a2)
            ), -0.5)))
        vab = torch.sum(torch.exp(torch.mul(torch.add(torch.div(torch.pow(
            mu_a1 - mu_b2, 2), var_a1 + var_b2), torch.log(var_a1 + var_b2)
            ), -0.5)))
        vbb = torch.sum(torch.exp(torch.mul(torch.add(torch.div(torch.pow(
            mu_b1 - mu_b2, 2), var_b1 + var_b2), torch.log(var_b1 + var_b2)
            ), -0.5)))
        loss = vaa + vbb - torch.mul(vab, 2.0)
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
