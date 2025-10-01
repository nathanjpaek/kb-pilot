import torch
from torch.nn.modules.loss import _Loss


class PSNR(_Loss):

    def __init__(self):
        super(PSNR, self).__init__()
        self.val_range = 255

    def _quantize(self, img):
        img = img * self.val_range
        img = img.clamp(0, self.val_range).round()
        return img

    def forward(self, x, y):
        diff = self._quantize(x) - self._quantize(y)
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)
        mse = diff.div(self.val_range).pow(2).view(n, -1).mean(dim=-1)
        psnr = -10 * mse.log10()
        return psnr.mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
