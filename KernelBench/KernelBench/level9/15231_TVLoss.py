import torch
from torch.nn import functional as F
import torch.nn as nn


def get_image_gradients(image: 'torch.Tensor', step: 'int'=1):
    """Returns image gradients (dy, dx) for each color channel, using
    the finite-difference approximation.
    Places the gradient [ie. I(x+1,y) - I(x,y)] on the base pixel (x, y).
    Both output tensors have the same shape as the input: [b, c, h, w].

    Arguments:
        image: Tensor with shape [b, c, h, w].
        step: the size of the step for the finite difference
    Returns:
        Pair of tensors (dy, dx) holding the vertical and horizontal
        image gradients (ie. 1-step finite difference). To match the
        original size image, for example with step=1, dy will always
        have zeros in the last row, and dx will always have zeros in
        the last column.
    """
    right = F.pad(image, (0, step, 0, 0))[..., :, step:]
    bottom = F.pad(image, (0, 0, 0, step))[..., step:, :]
    dx, dy = right - image, bottom - image
    dx[:, :, :, -step:] = 0
    dy[:, :, -step:, :] = 0
    return dx, dy


def get_outnorm(x: 'torch.Tensor', out_norm: 'str'='') ->torch.Tensor:
    """ Common function to get a loss normalization value. Can
        normalize by either the batch size ('b'), the number of
        channels ('c'), the image size ('i') or combinations
        ('bi', 'bci', etc)
    """
    img_shape = x.shape
    if not out_norm:
        return 1
    norm = 1
    if 'b' in out_norm:
        norm /= img_shape[0]
    if 'c' in out_norm:
        norm /= img_shape[-3]
    if 'i' in out_norm:
        norm /= img_shape[-1] * img_shape[-2]
    return norm


def get_4dim_image_gradients(image: 'torch.Tensor'):
    """Returns image gradients (dy, dx) for each color channel, using
    the finite-difference approximation.
    Similar to get_image_gradients(), but additionally calculates the
    gradients in the two diagonal directions: 'dp' (the positive
    diagonal: bottom left to top right) and 'dn' (the negative
    diagonal: top left to bottom right).
    Only 1-step finite difference has been tested and is available.

    Arguments:
        image: Tensor with shape [b, c, h, w].
    Returns:
        tensors (dy, dx, dp, dn) holding the vertical, horizontal and
        diagonal image gradients (1-step finite difference). dx will
        always have zeros in the last column, dy will always have zeros
        in the last row, dp will always have zeros in the last row.
    """
    right = F.pad(image, (0, 1, 0, 0))[..., :, 1:]
    bottom = F.pad(image, (0, 0, 0, 1))[..., 1:, :]
    botright = F.pad(image, (0, 1, 0, 1))[..., 1:, 1:]
    dx, dy = right - image, bottom - image
    dn, dp = botright - image, right - bottom
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0
    dp[:, :, -1, :] = 0
    return dx, dy, dp, dn


class TVLoss(nn.Module):
    """Calculate the L1 or L2 total variation regularization.
    Also can calculate experimental 4D directional total variation.
    Args:
        tv_type: regular 'tv' or 4D 'dtv'
        p: use the absolute values '1' or Euclidean distance '2' to
            calculate the tv. (alt names: 'l1' and 'l2')
        reduction: aggregate results per image either by their 'mean' or
            by the total 'sum'. Note: typically, 'sum' should be
            normalized with out_norm: 'bci', while 'mean' needs only 'b'.
        out_norm: normalizes the TV loss by either the batch size ('b'), the
            number of channels ('c'), the image size ('i') or combinations
            ('bi', 'bci', etc).
        beta: β factor to control the balance between sharp edges (1<β<2)
            and washed out results (penalizing edges) with β >= 2.
    Ref:
        Mahendran et al. https://arxiv.org/pdf/1412.0035.pdf
    """

    def __init__(self, tv_type: 'str'='tv', p=2, reduction: 'str'='mean',
        out_norm: 'str'='b', beta: 'int'=2) ->None:
        super(TVLoss, self).__init__()
        if isinstance(p, str):
            p = 1 if '1' in p else 2
        if p not in [1, 2]:
            raise ValueError(f'Expected p value to be 1 or 2, but got {p}')
        self.p = p
        self.tv_type = tv_type.lower()
        self.reduction = torch.sum if reduction == 'sum' else torch.mean
        self.out_norm = out_norm
        self.beta = beta

    def forward(self, x: 'torch.Tensor') ->torch.Tensor:
        norm = get_outnorm(x, self.out_norm)
        img_shape = x.shape
        if len(img_shape) == 3:
            reduce_axes = None
        elif len(img_shape) == 4:
            reduce_axes = -3, -2, -1
            x.size()[0]
        else:
            raise ValueError(
                f'Expected input tensor to be of ndim 3 or 4, but got {len(img_shape)}'
                )
        if self.tv_type in ('dtv', '4d'):
            gradients = get_4dim_image_gradients(x)
        else:
            gradients = get_image_gradients(x)
        loss = 0
        for grad_dir in gradients:
            if self.p == 1:
                loss += self.reduction(grad_dir.abs(), dim=reduce_axes)
            elif self.p == 2:
                loss += self.reduction(torch.pow(grad_dir, 2), dim=reduce_axes)
        loss = loss.sum() if 'b' in self.out_norm else loss.mean()
        if self.beta != 2:
            loss = torch.pow(loss, self.beta / 2)
        return loss * norm


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
