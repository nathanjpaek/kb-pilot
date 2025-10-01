import torch
import uuid
import torch.utils.data
import torch.utils.data.distributed
from torch import nn
import torch.nn.parallel
import torch.optim


def _get_right_parentheses_index_(struct_str):
    """get the position of the first right parenthese in string"""
    left_paren_count = 0
    for index, single_char in enumerate(struct_str):
        if single_char == '(':
            left_paren_count += 1
        elif single_char == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None


class PlainNetBasicBlockClass(nn.Module):
    """BasicBlock base class"""

    def __init__(self, in_channels=None, out_channels=None, stride=1,
        no_create=False, block_name=None, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.no_create = no_create
        self.block_name = block_name
        if self.block_name is None:
            self.block_name = f'uuid{uuid.uuid4().hex}'

    def forward(self, input_):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def __str__(self):
        return type(self
            ).__name__ + f'({self.in_channels},{self.out_channels},{self.stride})'

    def __repr__(self):
        return (type(self).__name__ +
            f'({self.block_name}|{self.in_channels},{self.out_channels},{self.stride})'
            )

    def get_output_resolution(self, input_resolution):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def get_FLOPs(self, input_resolution):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def get_model_size(self):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    def set_in_channels(self, channels):
        """subclass implementation"""
        raise RuntimeError('Not implemented')

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        """ class method

            :param s (str): basicblock str
            :return cls instance
        """
        assert PlainNetBasicBlockClass.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len(cls.__name__ + '('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        return cls(in_channels=in_channels, out_channels=out_channels,
            stride=stride, block_name=tmp_block_name, no_create=no_create
            ), struct_str[idx + 1:]

    @classmethod
    def is_instance_from_str(cls, struct_str):
        if struct_str.startswith(cls.__name__ + '(') and struct_str[-1] == ')':
            return True
        return False


class AdaptiveAvgPool(PlainNetBasicBlockClass):
    """Adaptive average pool layer"""

    def __init__(self, out_channels, output_size, no_create=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = out_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.no_create = no_create
        if not no_create:
            self.netblock = nn.AdaptiveAvgPool2d(output_size=(self.
                output_size, self.output_size))

    def forward(self, input_):
        return self.netblock(input_)

    def __str__(self):
        return (type(self).__name__ +
            f'({self.out_channels // self.output_size ** 2},{self.output_size})'
            )

    def __repr__(self):
        return (type(self).__name__ +
            f'({self.block_name}|{self.out_channels // self.output_size ** 2},                                        {self.output_size})'
            )

    def get_output_resolution(self, input_resolution):
        return self.output_size

    def get_FLOPs(self, input_resolution):
        return 0

    def get_model_size(self):
        return 0

    def set_in_channels(self, channels):
        self.in_channels = channels
        self.out_channels = channels

    @classmethod
    def create_from_str(cls, struct_str, no_create=False, **kwargs):
        assert AdaptiveAvgPool.is_instance_from_str(struct_str)
        idx = _get_right_parentheses_index_(struct_str)
        assert idx is not None
        param_str = struct_str[len('AdaptiveAvgPool('):idx]
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = f'uuid{uuid.uuid4().hex}'
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]
        param_str_split = param_str.split(',')
        out_channels = int(param_str_split[0])
        output_size = int(param_str_split[1])
        return AdaptiveAvgPool(out_channels=out_channels, output_size=
            output_size, block_name=tmp_block_name, no_create=no_create
            ), struct_str[idx + 1:]


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'out_channels': 4, 'output_size': 4}]
