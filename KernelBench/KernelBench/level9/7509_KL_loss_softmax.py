import torch
import torch.nn as nn
import torch.nn.init


class KL_loss_softmax(nn.Module):
    """
    Compute KL_divergence between all prediction score (already sum=1, omit softmax function)
    """

    def __init__(self):
        super(KL_loss_softmax, self).__init__()
        self.KL_loss = nn.KLDivLoss(reduce=False)

    def forward(self, im, s):
        img_prob = torch.log(im)
        s_prob = s
        KL_loss = self.KL_loss(img_prob, s_prob)
        loss = KL_loss.sum()
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
