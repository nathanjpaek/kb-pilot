import torch
import torch.utils.data
from torchvision.transforms import functional as F
import torch.nn as nn
import torch.nn.functional as F
from sklearn import *


class KLLoss(nn.Module):
    """
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \\alpha (1-softmax(x)[class])^gamma \\log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.

    """

    def __init__(self, margin=0.0, size_average=None, reduce=None,
        reduction='mean'):
        super(KLLoss, self).__init__()

    def forward(self, batch_exist, global_exist):
        KL_loss = F.kl_div(batch_exist.softmax(-1).log(), global_exist.
            softmax(-1).detach())
        return KL_loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
