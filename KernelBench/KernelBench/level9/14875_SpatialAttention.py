import torch
import torch.nn as nn
import torch.nn
import torch.utils.data


class SpatialAttention(nn.Module):

    def __init__(self, input_dim, context_dim):
        super().__init__()
        self.conv_context = nn.Conv2d(context_dim, input_dim, 1, stride=1,
            padding=0, bias=False)
        self.sm = nn.Softmax(dim=1)

    def forward(self, input, context, mask):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context.unsqueeze(3)
        sourceT = self.conv_context(sourceT).squeeze(3)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, queryL, -1).contiguous().view(
                batch_size * queryL, -1)
            attn = attn + mask.float() * -10000
        attn = self.sm(attn)
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.transpose(attn, 1, 2).contiguous()
        weightedContext = torch.bmm(sourceT, attn)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        attn = attn.view(batch_size, -1, ih, iw)
        return weightedContext, attn


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4])
        ]


def get_init_inputs():
    return [[], {'input_dim': 4, 'context_dim': 4}]
