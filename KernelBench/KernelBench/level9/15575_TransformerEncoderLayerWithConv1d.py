import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoderLayerWithConv1d(nn.Module):
    """
      Input and output shape: seqlen x batch_size x dim
    """

    def __init__(self, dim_model, nheads, dim_feedforward, dropout,
        kernel_size, stride):
        super(TransformerEncoderLayerWithConv1d, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(dim_model, nheads,
            dim_feedforward, dropout)
        self.conv1d = nn.Conv1d(dim_model, dim_model, kernel_size, stride=
            stride, padding=1)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = self.encoder_layer(src, src_mask, src_key_padding_mask)
        output = F.relu(self.conv1d(output.permute(1, 2, 0)))
        return output.permute(2, 0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim_model': 4, 'nheads': 4, 'dim_feedforward': 4,
        'dropout': 0.5, 'kernel_size': 4, 'stride': 1}]
