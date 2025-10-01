from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class PreActResPath(nn.Module):

    def __init__(self, in_features, config, super_block):
        super(PreActResPath, self).__init__()
        self.number_layers = config['num_layers']
        self.activate_dropout = True if config['activate_dropout'
            ] == 'Yes' else False
        self.activate_batch_norm = True if config['activate_batch_norm'
            ] == 'Yes' else False
        self.relu = nn.ReLU(inplace=True)
        if self.activate_batch_norm:
            setattr(self, 'b_norm_1', nn.BatchNorm1d(in_features))
        setattr(self, 'fc_1', nn.Linear(in_features, config[
            'num_units_%d_1' % super_block]))
        if self.activate_dropout:
            setattr(self, 'dropout_1', nn.Dropout(p=config['dropout_%d_1' %
                super_block]))
        for i in range(2, self.number_layers + 1):
            if self.activate_batch_norm:
                setattr(self, 'b_norm_%d' % i, nn.BatchNorm1d(config[
                    'num_units_%d_%d' % (super_block, i - 1)]))
            setattr(self, 'fc_%d' % i, nn.Linear(config['num_units_%d_%d' %
                (super_block, i - 1)], config['num_units_%d_%d' % (
                super_block, i)]))

    def forward(self, x):
        out = x
        for i in range(1, self.number_layers + 1):
            if self.activate_batch_norm:
                out = getattr(self, 'b_norm_%d' % i)(out)
            out = self.relu(out)
            out = getattr(self, 'fc_%d' % i)(out)
            if self.activate_dropout:
                out = getattr(self, 'dropout_%d' % i)(out)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'config': _mock_config(num_layers=1,
        activate_dropout=0.5, activate_batch_norm=4, num_units_4_1=4),
        'super_block': 4}]
