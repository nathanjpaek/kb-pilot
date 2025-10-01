import torch
import torch.nn as nn
import torch.onnx


class TransformerLayer(nn.Module):

    def __init__(self, channels, num_heads):
        super().__init__()
        self.q = nn.Linear(channels, channels, bias=False)
        self.k = nn.Linear(channels, channels, bias=False)
        self.v = nn.Linear(channels, channels, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=channels, num_heads=num_heads
            )
        self.fc1 = nn.Linear(channels, channels, bias=False)
        self.fc2 = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'channels': 4, 'num_heads': 4}]
