import torch
import torch.nn as nn


class PoseCriterion(nn.Module):

    def __init__(self, t_loss_fn=nn.MSELoss(), q_loss_fn=nn.MSELoss(), sax=
        0.0, saq=0.0, learn_beta=False):
        super(PoseCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        loss = torch.exp(-self.sax) * self.t_loss_fn(pred[:, :3], targ[:, :3]
            ) + self.sax + torch.exp(-self.saq) * self.q_loss_fn(pred[:, 3:
            ], targ[:, 3:]) + self.saq
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
