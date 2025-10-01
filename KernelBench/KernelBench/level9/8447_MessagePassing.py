import torch
import torch._C
import torch.serialization
from torch import nn
from torch.nn import Parameter


def make_onehot_kernel(kernel_size, index):
    """
    Make 2D one hot square kernel, i.e. h=w
    k[kernel_size, kernel_size] = 0 except k.view(-1)[index] = 1
    """
    kernel = torch.zeros(kernel_size, kernel_size)
    kernel.view(-1)[index] = 1
    return kernel.view(1, 1, kernel_size, kernel_size)


def make_spatial_kernel(kernel_size, bandwidth, isreshape=True):
    """
    Make 2D square smoothness kernel, i.e. h=w
    k = 1/bandwidth * exp(-(pj-pi)**2/(2*bandwidth**2))
    pj, pi = location of pixel
    """
    assert bandwidth > 0, 'bandwidth of kernel must be > 0'
    assert kernel_size % 2 != 0, 'kernel must be odd'
    p_end = (kernel_size - 1) // 2
    X = torch.linspace(-p_end, p_end, steps=kernel_size).expand(kernel_size,
        kernel_size)
    Y = X.clone().t()
    kernel = torch.exp(-(X ** 2 + Y ** 2) / (2 * bandwidth ** 2))
    kernel[p_end, p_end] = 0
    if isreshape:
        return kernel.view(1, 1, kernel_size, kernel_size)
    return kernel


class GaussianMask(nn.Module):
    """
    Break down Gaussian kernel (2nd part of appearance kernel) into CNN
    kj = (I(j) - I(i))**2/2*bandwidth**2, j#i
    but compute all maps instead of 1 kernel
    """

    def __init__(self, in_channels, kernel_size, bandwidth, iskernel=True):
        super(GaussianMask, self).__init__()
        assert bandwidth > 0, 'bandwidth of kernel must be > 0'
        assert kernel_size % 2 != 0, 'kernel must be odd'
        self.bandwidth = bandwidth
        self.iskernel = iskernel
        self.n_kernels = kernel_size ** 2 - 1
        kernel_weight = self._make_kernel_weight(in_channels, kernel_size,
            self.n_kernels)
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, in_channels * self.n_kernels,
            kernel_size, stride=1, padding=padding, groups=in_channels,
            bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(kernel_weight.view_as(self.conv.weight))

    def _make_kernel_weight(self, in_channels, kernel_size, n_kernels):
        kernel_weight = torch.zeros(in_channels, n_kernels, kernel_size,
            kernel_size)
        for i in range(n_kernels):
            index = i if i < n_kernels // 2 else i + 1
            kernel_i = make_onehot_kernel(kernel_size, index)
            kernel_weight[:, i, :] = kernel_i
        return kernel_weight

    def forward(self, X):
        batch_size, in_channels, H, W = X.shape
        Xj = self.conv(X).view(batch_size, in_channels, self.n_kernels, H, W)
        if not self.iskernel:
            return Xj
        Xi = X.unsqueeze(dim=2)
        K = (Xj - Xi) ** 2 / (2 * self.bandwidth ** 2)
        K = torch.exp(-K)
        return K


class SpatialFilter(nn.Module):
    """
    Break down spatial filter (smoothest kernel) into CNN blocks
    refer: https://arxiv.org/pdf/1210.5644.pdf
    """

    def __init__(self, n_classes, kernel_size, theta_gamma):
        super(SpatialFilter, self).__init__()
        padding = kernel_size // 2
        kernel_weight = make_spatial_kernel(kernel_size, theta_gamma)
        self.conv = nn.Conv2d(n_classes, n_classes, kernel_size, stride=1,
            padding=padding, groups=n_classes, bias=False)
        self.conv.weight.requires_grad = False
        self.conv.weight.copy_(kernel_weight)

    def forward(self, Q):
        Qtilde = self.conv(Q)
        norm_weight = self.conv(Q.new_ones(*Q.shape, requires_grad=False))
        Qtilde = Qtilde / norm_weight
        return Qtilde


class BilateralFilter(nn.Module):
    """
    Break down bilateral filter (appearance kernel) into CNN blocks
    remember that exp(-a-b) =exp(-a)*exp(b)
    """

    def __init__(self, in_channels, n_classes, kernel_size, theta_alpha,
        theta_beta):
        super(BilateralFilter, self).__init__()
        kernel_weight = make_spatial_kernel(kernel_size, theta_alpha,
            isreshape=False)
        self.spatial_weight = Parameter(kernel_weight[kernel_weight > 0].
            view(1, 1, 1, -1, 1, 1), requires_grad=False)
        self.gauss_mask_I = GaussianMask(in_channels, kernel_size, theta_beta)
        self.guass_mask_Q = GaussianMask(n_classes, kernel_size, 1,
            iskernel=False)

    def forward(self, Q, I):
        Ij = self.gauss_mask_I(I)
        Qj = self.guass_mask_Q(Q)
        Qj = Ij.unsqueeze(dim=2) * Qj.unsqueeze(dim=1)
        Qj = Qj * self.spatial_weight
        Qtilde = Qj.sum(dim=3)
        norm_weight = Ij * self.spatial_weight.squeeze(dim=2)
        norm_weight = norm_weight.sum(dim=2)
        Qtilde = Qtilde / norm_weight.unsqueeze(dim=2)
        return Qtilde


class MessagePassing(nn.Module):
    """
    Combine bilateral filter (appearance filter)
    and spatial filter to make message passing
    """

    def __init__(self, in_channels, n_classes, kernel_size=[3], theta_alpha
        =[2.0], theta_beta=[2.0], theta_gamma=[2.0]):
        super(MessagePassing, self).__init__()
        assert len(theta_alpha) == len(theta_beta
            ), 'theta_alpha and theta_beta have different lengths'
        self.n_bilaterals, self.n_spatials = len(theta_alpha), len(theta_gamma)
        for i in range(self.n_bilaterals):
            self.add_module('bilateral{}'.format(i), BilateralFilter(
                in_channels, n_classes, kernel_size[i], theta_alpha[i],
                theta_beta[i]))
        for i in range(self.n_spatials):
            self.add_module('spatial{}'.format(i), SpatialFilter(n_classes,
                kernel_size[i], theta_gamma[i]))

    def _get_child(self, child_name):
        return getattr(self, child_name)

    def forward(self, Q, I):
        filteredQ = []
        for i in range(self.n_bilaterals):
            tmp_bilateral = self._get_child('bilateral{}'.format(i))(Q, I)
            filteredQ.append(tmp_bilateral)
        for i in range(self.n_spatials):
            tmp_spatial = self._get_child('spatial{}'.format(i))(Q)
            filteredQ.append(tmp_spatial.unsqueeze(dim=1))
        Qtilde = torch.cat(filteredQ, dim=1)
        return Qtilde


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'n_classes': 4}]
