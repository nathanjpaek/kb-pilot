import torch
import torch.nn.functional as F


class MsgNorm(torch.nn.Module):

    def __init__(self, learn_msg_scale=False):
        super(MsgNorm, self).__init__()
        self.msg_scale = torch.nn.Parameter(torch.Tensor([1.0]),
            requires_grad=learn_msg_scale)

    def forward(self, x, msg, p=2):
        msg = F.normalize(msg, p=p, dim=1)
        x_norm = x.norm(p=p, dim=1, keepdim=True)
        msg = msg * x_norm * self.msg_scale
        return msg


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
