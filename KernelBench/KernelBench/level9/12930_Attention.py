import torch


class Attention(torch.nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hl, hr):
        hl = hl / hl.norm(dim=-1, keepdim=True)
        hr = hr / hr.norm(dim=-1, keepdim=True)
        a = (hl[:, None, :] * hr[None, :, :]).sum(dim=-1)
        mu_lr = hr - a.softmax(dim=1).transpose(1, 0) @ hl
        mu_rl = hl - a.softmax(dim=0) @ hr
        return mu_lr, mu_rl


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
