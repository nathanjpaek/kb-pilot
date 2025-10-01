import torch
import torch.nn as nn


class CAT_TokenEmbedding(nn.Module):

    def __init__(self, c_in=1, d_feature=10):
        super(CAT_TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_feature,
            kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in',
                    nonlinearity='leaky_relu')

    def forward(self, x: 'torch.Tensor'):
        x = x.unsqueeze(1)
        x = x.transpose(0, 2)
        x = self.tokenConv(x).permute(1, 2, 0)
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
