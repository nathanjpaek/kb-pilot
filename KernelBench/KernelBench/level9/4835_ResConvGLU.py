import math
import torch


class Conv1d(torch.nn.Conv1d):

    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity='relu')
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)


class Conv1d1x1(Conv1d):

    def __init__(self, in_channels, out_channels, bias):
        super(Conv1d1x1, self).__init__(in_channels=in_channels,
            out_channels=out_channels, kernel_size=1, padding=0, dilation=1,
            bias=bias)


class ResConvGLU(torch.nn.Module):

    def __init__(self, residual_channels, gate_channels, skip_channels,
        aux_channels, kernel_size, dilation=1, dropout=0.0, bias=True):
        super(ResConvGLU, self).__init__()
        self.layer_Dict = torch.nn.ModuleDict()
        self.layer_Dict['Conv1d'] = torch.nn.Sequential()
        self.layer_Dict['Conv1d'].add_module('Dropout', torch.nn.Dropout(p=
            dropout))
        self.layer_Dict['Conv1d'].add_module('Conv1d', Conv1d(in_channels=
            residual_channels, out_channels=gate_channels, kernel_size=
            kernel_size, padding=(kernel_size - 1) // 2 * dilation,
            dilation=dilation, bias=bias))
        self.layer_Dict['Aux'] = Conv1d1x1(in_channels=aux_channels,
            out_channels=gate_channels, bias=False)
        self.layer_Dict['Out'] = Conv1d1x1(in_channels=gate_channels // 2,
            out_channels=residual_channels, bias=bias)
        self.layer_Dict['Skip'] = Conv1d1x1(in_channels=gate_channels // 2,
            out_channels=skip_channels, bias=bias)

    def forward(self, audios, auxs):
        residuals = audios
        audios = self.layer_Dict['Conv1d'](audios)
        audios_Tanh, audios_Sigmoid = audios.split(audios.size(1) // 2, dim=1)
        auxs = self.layer_Dict['Aux'](auxs)
        auxs_Tanh, auxs_Sigmoid = auxs.split(auxs.size(1) // 2, dim=1)
        audios_Tanh = torch.tanh(audios_Tanh + auxs_Tanh)
        audios_Sigmoid = torch.sigmoid(audios_Sigmoid + auxs_Sigmoid)
        audios = audios_Tanh * audios_Sigmoid
        outs = (self.layer_Dict['Out'](audios) + residuals) * math.sqrt(0.5)
        skips = self.layer_Dict['Skip'](audios)
        return outs, skips


def get_inputs():
    return [torch.rand([4, 4, 2]), torch.rand([4, 4, 2])]


def get_init_inputs():
    return [[], {'residual_channels': 4, 'gate_channels': 4,
        'skip_channels': 4, 'aux_channels': 4, 'kernel_size': 4}]
