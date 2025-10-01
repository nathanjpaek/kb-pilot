import torch


class BSConvU(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
        padding=1, dilation=1, bias=True, padding_mode='zeros', with_norm=
        True, bn_kwargs=None):
        super().__init__()
        self.with_norm = with_norm
        if bn_kwargs is None:
            bn_kwargs = {}
        self.pw = torch.nn.Conv2d(in_channels=in_channels, out_channels=
            out_channels, kernel_size=(1, 1), stride=1, padding=0, dilation
            =1, groups=1, bias=False)
        if with_norm:
            self.ln = torch.nn.LayerNorm(out_channels, **bn_kwargs)
        self.dw = torch.nn.Conv2d(in_channels=out_channels, out_channels=
            out_channels, kernel_size=kernel_size, stride=stride, padding=
            padding, dilation=dilation, groups=out_channels, bias=bias,
            padding_mode=padding_mode)

    def forward(self, fea):
        fea = self.pw(fea)
        if self.with_norm:
            fea = self.ln(fea.permute(0, 2, 3, 1))
        fea = self.dw(fea.permute(0, 3, 1, 2))
        return fea


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4}]
