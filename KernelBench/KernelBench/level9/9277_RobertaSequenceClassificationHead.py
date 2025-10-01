import torch
import torch.nn as nn
import torch.utils.data
import torch.onnx.operators
import torch.optim
import torch.optim.lr_scheduler


class RobertaSequenceClassificationHead(nn.Module):
    """Head for sequence-level classification tasks. Ignores the <s> vector."""

    def __init__(self, input_dim, inner_dim, kernel_size, num_classes,
        pooler_dropout):
        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=input_dim, out_channels=
            inner_dim, kernel_size=kernel_size)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = torch.transpose(features, 1, 2)
        x = self.conv_layer(x)
        x = torch.max(x, dim=2).values
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4, 'inner_dim': 4, 'kernel_size': 4,
        'num_classes': 4, 'pooler_dropout': 0.5}]
