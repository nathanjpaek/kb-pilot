import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    """
    MLP based decoder model for edge prediction.
    """

    def __init__(self, input_dim, num_classes, dropout=0.0, bias=False,
        init='xav_uniform'):
        super(MLPDecoder, self).__init__()
        assert init in ('xav_uniform', 'kaiming_uniform', 'xav_normal',
            'kaiming_normal')
        self.weight = nn.Parameter(torch.empty(input_dim, num_classes if 
            num_classes > 2 else 1))
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(num_classes if num_classes >
                2 else 1))
        self.dropout = nn.Dropout(p=dropout)
        if init == 'xav_uniform':
            torch.nn.init.xavier_uniform_(self.weight.data)
        elif init == 'kaiming_uniform':
            torch.nn.init.kaiming_uniform_(self.weight.data)
        elif init == 'xav_normal':
            torch.nn.init.xavier_normal_(self.weight.data)
        else:
            torch.nn.init.kaiming_normal_(self.weight.data)
        if bias:
            torch.nn.init.zeros_(self.bias.data)

    def forward(self, input, r_indices, c_indices):
        x = self.dropout(input).float()
        start_inputs = x[r_indices]
        end_inputs = x[c_indices]
        diff = torch.abs(start_inputs - end_inputs)
        out = torch.matmul(diff, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64),
        torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'input_dim': 4, 'num_classes': 4}]
