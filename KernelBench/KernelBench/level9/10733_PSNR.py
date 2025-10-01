import torch
import torch as th
import torch.utils.data


class PSNR(th.nn.Module):

    def __init__(self):
        super(PSNR, self).__init__()
        self.mse = th.nn.MSELoss()

    def forward(self, out, ref):
        mse = self.mse(out, ref)
        return -10 * th.log10(mse + 1e-12)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
