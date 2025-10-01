import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.onnx.operators


class Attention(nn.Module):

    def __init__(self, input_dim, source_dim=None, output_dim=None, bias=False
        ):
        super(Attention, self).__init__()
        if source_dim is None:
            source_dim = input_dim
        if output_dim is None:
            output_dim = input_dim
        self.input_dim = input_dim
        self.source_dim = source_dim
        self.output_dim = output_dim
        self.input_proj = nn.Linear(input_dim, source_dim, bias=bias)
        self.output_proj = nn.Linear(input_dim + source_dim, output_dim,
            bias=bias)

    def forward(self, input, source_hids, mask=None):
        batch_size = input.size(0)
        source_len = source_hids.size(1)
        x = self.input_proj(input)
        attn = torch.bmm(x, source_hids.transpose(1, 2))
        if mask is not None:
            attn.data.masked_fill_(mask, -float('inf'))
        attn = F.softmax(attn.view(-1, source_len), dim=1).view(batch_size,
            -1, source_len)
        mix = torch.bmm(attn, source_hids)
        combined = torch.cat((mix, input), dim=2)
        output = torch.tanh(self.output_proj(combined.view(-1, self.
            input_dim + self.source_dim))).view(batch_size, -1, self.output_dim
            )
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'input_dim': 4}]
