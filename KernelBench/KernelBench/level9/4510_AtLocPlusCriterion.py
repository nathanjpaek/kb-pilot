import torch
import torch.nn as nn
import torch.nn.init


def calc_vos_simple(poses):
    vos = []
    for p in poses:
        pvos = [(p[i + 1].unsqueeze(0) - p[i].unsqueeze(0)) for i in range(
            len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


class AtLocPlusCriterion(nn.Module):

    def __init__(self, t_loss_fn=nn.L1Loss(), q_loss_fn=nn.L1Loss(), sax=
        0.0, saq=0.0, srx=0.0, srq=0.0, learn_beta=False, learn_gamma=False):
        super(AtLocPlusCriterion, self).__init__()
        self.t_loss_fn = t_loss_fn
        self.q_loss_fn = q_loss_fn
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)
        self.srx = nn.Parameter(torch.Tensor([srx]), requires_grad=learn_gamma)
        self.srq = nn.Parameter(torch.Tensor([srq]), requires_grad=learn_gamma)

    def forward(self, pred, targ):
        s = pred.size()
        abs_loss = torch.exp(-self.sax) * self.t_loss_fn(pred.view(-1, *s[2
            :])[:, :3], targ.view(-1, *s[2:])[:, :3]) + self.sax + torch.exp(
            -self.saq) * self.q_loss_fn(pred.view(-1, *s[2:])[:, 3:], targ.
            view(-1, *s[2:])[:, 3:]) + self.saq
        pred_vos = calc_vos_simple(pred)
        targ_vos = calc_vos_simple(targ)
        s = pred_vos.size()
        vo_loss = torch.exp(-self.srx) * self.t_loss_fn(pred_vos.view(-1, *
            s[2:])[:, :3], targ_vos.view(-1, *s[2:])[:, :3]
            ) + self.srx + torch.exp(-self.srq) * self.q_loss_fn(pred_vos.
            view(-1, *s[2:])[:, 3:], targ_vos.view(-1, *s[2:])[:, 3:]
            ) + self.srq
        loss = abs_loss + vo_loss
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
