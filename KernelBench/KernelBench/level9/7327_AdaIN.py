import torch


class AdaIN(torch.nn.Module):

    def __init__(self, channels_in, channels_out, norm=True):
        super(AdaIN, self).__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.norm = norm
        self.affine_scale = torch.nn.Linear(channels_in, channels_out, bias
            =True)
        self.affine_bias = torch.nn.Linear(channels_in, channels_out, bias=True
            )

    def forward(self, x, w):
        ys = self.affine_scale(w)
        yb = self.affine_bias(w)
        ys = torch.unsqueeze(ys, -1)
        yb = torch.unsqueeze(yb, -1)
        xm = torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))
        if self.norm:
            xm_mean = torch.mean(xm, dim=2, keepdims=True)
            xm_centered = xm - xm_mean
            xm_std_rev = torch.rsqrt(torch.mean(torch.mul(xm_centered,
                xm_centered), dim=2, keepdims=True))
            xm_norm = xm_centered - xm_std_rev
        else:
            xm_norm = xm
        xm_scaled = xm_norm * ys + yb
        return torch.reshape(xm_scaled, x.shape)


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels_in': 4, 'channels_out': 4}]
