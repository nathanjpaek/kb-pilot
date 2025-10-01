import torch
import torch.nn as nn
import torch.nn.parallel


class Memory(nn.Module):

    def __init__(self):
        super(Memory, self).__init__()
        self.sm = nn.Softmax()
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask

    def forward(self, input, context_key, content_value):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x idf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context_key.size(0), context_key.size(2)
        target = input.view(batch_size, -1, queryL)
        targetT = torch.transpose(target, 1, 2).contiguous()
        sourceT = context_key
        weight = torch.bmm(targetT, sourceT)
        weight = weight.view(batch_size * queryL, sourceL)
        if self.mask is not None:
            mask = self.mask.repeat(queryL, 1)
            weight.data.masked_fill_(mask.data, -float('inf'))
        weight = torch.nn.functional.softmax(weight, dim=1)
        weight = weight.view(batch_size, queryL, sourceL)
        weight = torch.transpose(weight, 1, 2).contiguous()
        weightedContext = torch.bmm(content_value, weight)
        weightedContext = weightedContext.view(batch_size, -1, ih, iw)
        weight = weight.view(batch_size, -1, ih, iw)
        return weightedContext, weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4,
        4, 4])]


def get_init_inputs():
    return [[], {}]
