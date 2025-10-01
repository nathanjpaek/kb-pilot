import torch


def psnr(gt, pred, data_range=None, batch=True, reduce=True):
    """ Compute the peak signal to noise ratio (psnr)
    :param gt: gt image (torch.Tensor
    :param pred: input image (torch.Tensor)
    :param data_range: if None, estimated from gt
    :return: (mean) psnr
    """
    if batch:
        batch_size = gt.shape[0]
    else:
        batch_size = 1
    pred = pred.contiguous().view(batch_size, -1)
    gt = gt.contiguous().view(batch_size, -1)
    if data_range is None:
        data_range = gt.max(dim=1)[0]
    mse_err = (abs(gt - pred) ** 2).mean(1)
    psnr_val = 10 * torch.log10(data_range ** 2 / mse_err)
    if reduce:
        return psnr_val.mean()
    else:
        return psnr_val


class PSNRLoss(torch.nn.Module):
    """
    Computes PSNR between two images according to:

    psnr(x, y) = 10 * log10(1/MSE(x, y)), MSE(x, y) = ||x-y||^2 / size(x)

    Parameters:
    -----------

    x: Tensor - gterence image (or batch)
    y: Tensor - reconstructed image (or batch)
    normalized: bool - If abs(data) is normalized to [0, 1]
    batch_mode: bool - If batch is passed, set this to True
    is_complex: bool - If data is complex valued, 2 values (e.g. (x,y)) are paired

    Notice that ``abs'' squares
    Be cagtul with the order, since peak intensity is taken from the gterence
    image (taking from reconstruction yields a different value).

    """

    def __init__(self, batch=True, reduce=True):
        """
        normalized: bool - If abs(data) is normalized to [0, 1]
        batch_mode: bool - If batch is passed, set this to True
        is_complex: bool - If data is complex valued, 2 values (e.g. (x,y)) are paired
        """
        super(PSNRLoss, self).__init__()
        self.batch = batch
        self.reduce = reduce

    def forward(self, pred, gt, data_range=None):
        return psnr(pred, gt, data_range=data_range, batch=self.batch,
            reduce=self.reduce)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
