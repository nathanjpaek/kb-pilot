import torch
import torch.optim


def _angular_error(gt: 'torch.Tensor', pred: 'torch.Tensor', radians: 'bool'):
    relative = gt @ torch.transpose(pred, -2, -1)
    trace = relative[:, 0, 0] + relative[:, 1, 1] + relative[:, 2, 2]
    trace = torch.clamp(trace, -1.0, 3.0)
    phi = 0.5 * (trace - 1.0)
    return phi.acos() if radians else torch.rad2deg(phi.acos())


class AngleError(torch.nn.Module):

    def __init__(self, radians: 'bool'=True):
        super(AngleError, self).__init__()
        self.radians = radians

    def forward(self, gt: 'torch.Tensor', pred: 'torch.Tensor') ->torch.Tensor:
        return _angular_error(gt, pred, self.radians).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
