import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda
import torch.distributed


class MConn(nn.Module):
    """ My custom connection module
    """

    def __init__(self, _dim_1, _dim_2, _dim_3, _linear=False, _ln_size=None):
        super(MConn, self).__init__()
        self.linear1 = nn.Linear(_dim_1, _dim_2)
        self.linear2 = nn.Linear(_dim_2, _dim_3)
        if not _linear:
            self.linearw = nn.Linear(_dim_1, _dim_3)
            self.USE_RS = True
        else:
            self.USE_RS = False
        if _ln_size is not None:
            self.layer_norm = nn.LayerNorm(_ln_size)
            self.USE_LN = True
        else:
            self.USE_LN = False

    def forward(self, _input):
        _output = self.linear2(F.leaky_relu(F.dropout(self.linear1(_input),
            p=0.3), inplace=True))
        if self.USE_RS:
            output = F.dropout(self.linearw(_input), p=0.3)
            return self.layer_norm(_output + output
                ) if self.USE_LN else _output + output
        else:
            return self.layer_norm(_output) if self.USE_LN else _output


class MConnBlock(nn.Module):
    """  My custom connection block module
    """

    def __init__(self, _dim_1, _dim_2, _dim_3, _linear=False, _ln_size=None):
        super(MConnBlock, self).__init__()
        _mid_ln_size = (_ln_size[0], _dim_2) if _ln_size else None
        self.MConn1 = MConn(_dim_1, _dim_2, _dim_2, _linear, _mid_ln_size)
        self.MConn6 = MConn(_dim_2, _dim_2, _dim_2, _linear, _mid_ln_size)
        self.MConn7 = MConn(_dim_2, _dim_2, _dim_3, _linear, _ln_size)

    def forward(self, _input):
        _output = self.MConn1(_input)
        _output = self.MConn6(_output)
        _output = self.MConn7(_output)
        return _output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'_dim_1': 4, '_dim_2': 4, '_dim_3': 4}]
