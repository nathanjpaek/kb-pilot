import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as F
from torch.nn import functional as F
import torch.utils.data


class NonLocal2d(nn.Module):

    def __init__(self, dim_in, dim_inner, dim_out, max_pool_stride=2,
        use_maxpool=True, use_gn=False, use_scale=True):
        super().__init__()
        self.dim_inner = dim_inner
        self.use_maxpool = use_maxpool
        self.use_gn = use_gn
        self.use_scale = use_scale
        self.theta = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        if self.use_maxpool:
            self.pool = nn.MaxPool2d(kernel_size=max_pool_stride, stride=
                max_pool_stride, padding=0)
        self.phi = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.g = nn.Conv2d(dim_in, dim_inner, 1, stride=1, padding=0)
        self.out = nn.Conv2d(dim_inner, dim_out, 1, stride=1, padding=0)
        if self.use_gn:
            self.gn = nn.GroupNorm(32, dim_out, eps=1e-05)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.dim_inner, -1)
        theta_x = theta_x.permute(0, 2, 1)
        if self.use_maxpool:
            pool_x = self.pool(x)
        else:
            pool_x = x
        phi_x = self.phi(pool_x).view(batch_size, self.dim_inner, -1)
        g_x = self.g(pool_x).view(batch_size, self.dim_inner, -1)
        theta_phi = torch.matmul(theta_x, phi_x)
        if self.use_scale:
            theta_phi_sc = theta_phi * self.dim_inner ** -0.5
        else:
            theta_phi_sc = theta_phi
        p_x = F.softmax(theta_phi_sc, dim=-1)
        p_x = p_x.permute(0, 2, 1)
        t_x = torch.matmul(g_x, p_x)
        t_x = t_x.view(batch_size, self.dim_inner, *x.size()[2:])
        y = self.out(t_x)
        if self.use_gn:
            y = self.gn(y)
        return y + x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_in': 4, 'dim_inner': 4, 'dim_out': 4}]
