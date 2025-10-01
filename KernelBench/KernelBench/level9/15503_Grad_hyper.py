import torch
import torch.nn.functional


class Grad_hyper(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1'):
        super(Grad_hyper, self).__init__()
        self.penalty = penalty

    def forward(self, y_pred, wts):
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx, dim=[1, 2, 3]) + torch.mean(dy, dim=[1, 2, 3])
        grad = d / 2.0 * wts
        None
        return torch.mean(grad)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
