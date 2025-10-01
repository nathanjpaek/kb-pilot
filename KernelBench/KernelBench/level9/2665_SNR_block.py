import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class ChannelGate_sub(nn.Module):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(self, in_channels, num_gates=None, return_gates=False,
        gate_activation='sigmoid', reduction=16, layer_norm=False):
        super(ChannelGate_sub, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.return_gates = return_gates
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction,
            kernel_size=1, bias=True, padding=0)
        self.norm1 = None
        if layer_norm:
            self.norm1 = nn.LayerNorm((in_channels // reduction, 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, num_gates,
            kernel_size=1, bias=True, padding=0)
        if gate_activation == 'sigmoid':
            self.gate_activation = nn.Sigmoid()
        elif gate_activation == 'relu':
            self.gate_activation = nn.ReLU(inplace=True)
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError('Unknown gate activation: {}'.format(
                gate_activation))

    def forward(self, x):
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x, input * (1 - x), x


class SNR_block(nn.Module):
    """A mini-network for the SNR module
	
	1. Instance normalization
	2. Channel attention

	"""

    def __init__(self, in_channels, num_classes=2):
        super(SNR_block, self).__init__()
        self.IN = nn.InstanceNorm2d(in_channels, affine=True)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.style_reid_laye = ChannelGate_sub(in_channels, num_gates=
            in_channels, return_gates=False, gate_activation='sigmoid',
            reduction=4, layer_norm=False)

    def forward(self, x):
        x_IN = self.IN(x)
        x_style = x - x_IN
        (x_style_reid_useful, x_style_reid_useless, _selective_weight_useful
            ) = self.style_reid_laye(x_style)
        x = x_IN + x_style_reid_useful
        x_useless = x_IN + x_style_reid_useless
        return x_IN, x, x_useless, F.softmax(self.fc(self.global_avgpool(
            x_IN).view(x_IN.size(0), -1))), F.softmax(self.fc(self.
            global_avgpool(x).view(x.size(0), -1))), F.softmax(self.fc(self
            .global_avgpool(x_useless).view(x_useless.size(0), -1))), self.fc(
            self.global_avgpool(x_IN).view(x_IN.size(0), -1)), self.fc(self
            .global_avgpool(x).view(x.size(0), -1)), self.fc(self.
            global_avgpool(x_useless).view(x_useless.size(0), -1))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4}]
