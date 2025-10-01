import torch
from torch import nn


class TotalVariationLoss(nn.Module):

    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """
        h_shift = torch.pow(img[0, :, :, :-1] - img[0, :, :, 1:], 2)
        w_shift = torch.pow(img[0, :, :-1, :] - img[0, :, 1:, :], 2)
        loss = tv_weight * (torch.sum(h_shift) + torch.sum(w_shift))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
