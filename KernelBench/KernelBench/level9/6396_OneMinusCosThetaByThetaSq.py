import torch
from torch import cos
from torch import sin


def get_small_and_large_angle_inds(theta: 'torch.Tensor', eps: 'float'=0.001):
    """Returns the indices of small and non-small (large) angles, given
    a tensor of angles, and the threshold below (exclusive) which angles
    are considered 'small'.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.
    """
    small_inds = torch.abs(theta) < eps
    large_inds = small_inds == 0
    return small_inds, large_inds


def grad_one_minus_cos_theta_by_theta_sq(theta: 'torch.Tensor', eps:
    'float'=0.001):
    """Computes :math:`\\frac{\\partial}{\\partial \\theta}\\frac{1 - cos \\theta}{\\theta^2}`.

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta ** 2
    result[s] = -theta[s] / 12 * (1 - theta_sq[s] / 5 * (1 / 3 - theta_sq[s
        ] / 56 * (1 / 2 - theta_sq[s] / 135)))
    result[l] = sin(theta[l]) / theta_sq[l] - 2 * (1 - cos(theta[l])) / (
        theta_sq[l] * theta[l])
    return result


def one_minus_cos_theta_by_theta_sq(theta: 'torch.Tensor', eps: 'float'=0.001):
    """Computes :math:`\\frac{1 - cos \\theta}{\\theta^2}`. 

    Args:
        theta (torch.Tensor): Angle (magnitude of axis-angle vector).
        eps (float): Threshold (exclusive) below which an angle is
            considered 'small'.

    """
    result = torch.zeros_like(theta)
    s, l = get_small_and_large_angle_inds(theta, eps)
    theta_sq = theta ** 2
    result[s] = 1 / 2 * (1 - theta_sq[s] / 12 * (1 - theta_sq[s] / 30 * (1 -
        theta_sq[s] / 56)))
    result[l] = (1 - cos(theta[l])) / theta_sq[l]
    return result


class OneMinusCosThetaByThetaSq_Function(torch.autograd.Function):

    @staticmethod
    def forward(ctx, theta):
        ctx.save_for_backward(theta)
        return one_minus_cos_theta_by_theta_sq(theta)

    @staticmethod
    def backward(ctx, grad_output):
        theta, = ctx.saved_tensors
        grad_theta = None
        if ctx.needs_input_grad[0]:
            grad_theta = grad_output * grad_one_minus_cos_theta_by_theta_sq(
                theta)
        return grad_theta


class OneMinusCosThetaByThetaSq(torch.nn.Module):

    def __init__(self):
        super(OneMinusCosThetaByThetaSq, self).__init__()

    def forward(self, x):
        return OneMinusCosThetaByThetaSq_Function.apply(x)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
