import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.onnx
import torch.nn.parallel


class Attention(nn.Module):

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.linear_out = nn.Linear(dim * 2, dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)
        attn = torch.bmm(output, context.transpose(1, 2))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size,
            -1, input_size)
        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))
            ).view(batch_size, -1, hidden_size)
        return output, attn


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'dim': 4}]
