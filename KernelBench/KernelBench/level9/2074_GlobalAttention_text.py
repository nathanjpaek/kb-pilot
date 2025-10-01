import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data


class GlobalAttention_text(nn.Module):

    def __init__(self, idf, cdf):
        super(GlobalAttention_text, self).__init__()
        self.conv_context = nn.Conv1d(cdf, idf, kernel_size=1, stride=1,
            padding=0)
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = self.conv_context(context)
        attn = torch.bmm(targetT, sourceT)
        attn = attn.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            attn.data.masked_fill_(mask.data, -float('inf'))
        attn = attn.view(batch_size, queryL, sourceL)
        attn = torch.nn.Softmax(dim=1)(attn)
        text_weighted = torch.bmm(target, attn)
        return text_weighted


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'idf': 4, 'cdf': 4}]
