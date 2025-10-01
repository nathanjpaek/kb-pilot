import logging
import torch
import numpy as np
import torch.nn as nn


class Concat(nn.Module):

    def __init__(self, args=None):
        super(Concat, self).__init__()
        self.index = -1
        self.verbose = print
        self.enable = False
        self.input_index = ''
        self.tag = 'fm'
        self.args = args
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root'
                ) else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def update_concat_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index = parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))
        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or isinstance(by_index, str
                ) and by_index != 'all':
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError):
                    self.verbose('unexpect string in by_index: {}'.format(
                        by_index))
            if by_index == 'all' or self.index in by_index:
                if 'by_tag' in parameters and self.tag in parameters['by_tag'
                    ] or 'by_tag' not in parameters:
                    for k, v in list(parameters.items()):
                        if hasattr(self, '{}'.format(k)):
                            if isinstance(v, str):
                                v = v.replace("'", '').replace('"', '')
                            if isinstance(getattr(self, k), bool):
                                v = False if v in ['False', 'false', False
                                    ] else True
                            elif isinstance(getattr(self, k), int):
                                v = int(v)
                            if not isinstance(getattr(self, k), torch.Tensor):
                                setattr(self, '{}'.format(k), v)
                                self.verbose('update {}_{} to {} for index {}'
                                    .format(self.tag, k, getattr(self, k,
                                    'Non-Exist'), self.index))

    def __repr__(self):
        base = super(Concat, self).__repr__()
        if self.enable:
            base = base + '-index({})-input({})'.format(self.index, self.
                input_index)
        return base

    def forward(self, x, y):
        _N, _C, _H, _W = x.shape
        if self.enable:
            input_index = self.input_index.split('/')
            scale = []
            for i in input_index:
                if i in self.args.global_buffer:
                    scale = scale + self.args.global_buffer[i].tolist()
                else:
                    self.verbose('warning {} not found in global_buffer'.
                        format(i))
            scaled = np.array(scale)
            scaled = scaled.reshape(-1)
            self.args.global_buffer['concat-{}'.format(self.index)] = scaled
            self.verbose('add concat-{} to global_buffer'.format(self.index))
        return torch.cat((x, y), dim=1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
