import torch
import torch as t
import torch.nn as nn
from abc import ABC
from torch.nn.utils.weight_norm import weight_norm


def conv1x1(in_planes, out_planes, stride=1):
    """
    Create a 1x1 2d convolution block
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
        bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    """
    Create a 3x3 2d convolution block
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
        padding=1, bias=False)


class NeuralNetworkModule(nn.Module, ABC):
    """
    Note: input device and output device are determined by module parameters,
          your input module / output submodule should not store parameters on
          more than one device, and you also should not move your output to
          other devices other than your parameter storage device in forward().
    """

    def __init__(self):
        super().__init__()
        self.input_module = None
        self.output_module = None

    def set_input_module(self, input_module: 'nn.Module'):
        """
        Set the input submodule of current module.
        """
        self.input_module = input_module
        if not isinstance(input_module, NeuralNetworkModule):
            if isinstance(input_module, nn.Sequential):
                input_module = self.find_child(input_module, True)
            if len({p.device for p in input_module.parameters()}) > 1:
                raise RuntimeError(
                    'Input module must be another NeuralNetworkModule or locate on one single device.'
                    )

    def set_output_module(self, output_module: 'nn.Module'):
        """
        Set the output submodule of current module.
        """
        self.output_module = output_module
        if not isinstance(output_module, NeuralNetworkModule):
            if isinstance(output_module, nn.Sequential):
                output_module = self.find_child(output_module, False)
            if len({p.device for p in output_module.parameters()}) > 1:
                raise RuntimeError(
                    'Output module must be another NeuralNetworkModule or locate on one single device.'
                    )

    @property
    def input_device(self):
        if self.input_module is None:
            raise RuntimeError('Input module not set.')
        elif not isinstance(self.input_module, NeuralNetworkModule):
            dev_set = {p.device for p in self.input_module.parameters()}
            if len(dev_set) != 1:
                raise RuntimeError(
                    'This input module contains parameters on different devices, please consider about splitting it.'
                    )
            else:
                return list(dev_set)[0]
        else:
            return self.input_module.input_device

    @property
    def output_device(self):
        if self.output_module is None and self.input_module is None:
            raise RuntimeError('Output module not set.')
        elif self.output_module is not None:
            if not isinstance(self.output_module, NeuralNetworkModule):
                dev_set = {p.device for p in self.output_module.parameters()}
                if len(dev_set) != 1:
                    raise RuntimeError(
                        'This output module contains parameters on different devices, please consider about splitting it.'
                        )
                else:
                    return list(dev_set)[0]
            else:
                return self.output_module.output_device
        else:
            return self.input_device

    @staticmethod
    def find_child(seq, is_first=True):
        """
        Find the first / last leaf child module.
        """
        if isinstance(seq, nn.Sequential):
            if is_first:
                return NeuralNetworkModule.find_child(seq[0], is_first)
            else:
                return NeuralNetworkModule.find_child(seq[-1], is_first)
        else:
            return seq

    def forward(self, *_):
        pass


class BasicBlockWN(NeuralNetworkModule):
    """
    Basic block with weight normalization
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, stride=1, **__):
        """
        Create a basic block of resnet.

        Args:
            in_planes:  Number of input planes.
            out_planes: Number of output planes.
            stride:     Stride of convolution.
        """
        super().__init__()
        self.conv1 = weight_norm(conv3x3(in_planes, out_planes, stride))
        self.conv2 = weight_norm(conv3x3(out_planes, self.expansion *
            out_planes))
        self.shortcut = nn.Sequential()
        self.set_input_module(self.conv1)
        if stride != 1 or in_planes != self.expansion * out_planes:
            self.shortcut = nn.Sequential(weight_norm(conv1x1(in_planes, 
                self.expansion * out_planes, stride)))

    def forward(self, x):
        out = t.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = t.relu(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_planes': 4, 'out_planes': 4}]
