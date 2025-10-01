import torch


class Downsample(torch.nn.Module):

    def __init__(self, s, use_max=False, batch_mode=False):
        super(Downsample, self).__init__()
        self.batch_mode = batch_mode
        if use_max:
            layer = torch.nn.MaxPool3d(s, stride=s)
        else:
            layer = torch.nn.Conv3d(1, 1, s, stride=s)
            layer.weight.data.fill_(1.0 / layer.weight.data.nelement())
            layer.bias.data.fill_(0)
        self.layer = layer

    def forward(self, vol):
        if self.batch_mode:
            out_vol = self.layer.forward(vol)
        else:
            out_vol = self.layer.forward(torch.unsqueeze(torch.unsqueeze(
                vol, 0), 0))[0, 0]
        return out_vol


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'s': 4}]
