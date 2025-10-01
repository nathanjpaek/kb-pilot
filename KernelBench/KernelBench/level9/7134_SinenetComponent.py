import torch


class SinenetComponent(torch.nn.Module):

    def __init__(self, time_len, i):
        super().__init__()
        self.time_len = time_len
        self.i = i
        self.t_wav = 1.0 / 16000
        self.log_f_mean = 5.02654
        self.log_f_std = 0.373288
        self.a = torch.nn.Parameter(torch.Tensor(1))
        self.phi = torch.nn.Parameter(torch.Tensor(1))

    def forward(self, x, f, t):
        i_f = torch.mul(self.i, f)
        i_f_t = torch.mul(i_f, t)
        deg = torch.add(i_f_t, self.phi)
        s = torch.sin(deg)
        self.W = torch.mul(self.a, s)
        h_SBT = torch.mul(self.W, x)
        h_SB = torch.sum(h_SBT, dim=-1, keepdim=False)
        return h_SB


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'time_len': 4, 'i': 4}]
