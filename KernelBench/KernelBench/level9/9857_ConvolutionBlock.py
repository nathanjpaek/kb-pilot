import torch


class BaseModule(torch.nn.Module):

    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def nparams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class Conv1dWithInitialization(BaseModule):

    def __init__(self, **kwargs):
        super(Conv1dWithInitialization, self).__init__()
        self.conv1d = torch.nn.Conv1d(**kwargs)
        torch.nn.init.orthogonal_(self.conv1d.weight.data, gain=1)

    def forward(self, x):
        return self.conv1d(x)


class ConvolutionBlock(BaseModule):

    def __init__(self, in_channels, out_channels, dilation):
        super(ConvolutionBlock, self).__init__()
        self.leaky_relu = torch.nn.LeakyReLU(0.2)
        self.convolution = Conv1dWithInitialization(in_channels=in_channels,
            out_channels=out_channels, kernel_size=3, stride=1, padding=
            dilation, dilation=dilation)

    def forward(self, x):
        outputs = self.leaky_relu(x)
        outputs = self.convolution(outputs)
        return outputs


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'out_channels': 4, 'dilation': 1}]
