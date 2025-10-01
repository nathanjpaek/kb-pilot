import math
import torch


class BinaryChunk(torch.nn.Module):

    def __init__(self, nCls, isLogit=False, pooling='max', chunk_dim=-1):
        super(BinaryChunk, self).__init__()
        self.nClass = nCls
        self.nChunk = int(math.ceil(math.log2(self.nClass)))
        self.pooling = pooling
        self.isLogit = isLogit

    def __repr__(self):
        main_str = super(BinaryChunk, self).__repr__()
        if self.isLogit:
            main_str += '_logit'
        main_str += (
            f'_nChunk{self.nChunk}_cls[{self.nClass}]_pool[{self.pooling}]')
        return main_str

    def chunk_poll(self, ck, nSamp):
        x2 = ck.contiguous().view(nSamp, -1)
        if self.pooling == 'max':
            x3 = torch.max(x2, 1)
            return x3.values
        else:
            x3 = torch.mean(x2, 1)
            return x3

    def forward(self, x):
        nSamp = x.shape[0]
        x_max = []
        for ck in x.chunk(self.nChunk, -1):
            if self.isLogit:
                x_max.append(self.chunk_poll(ck, nSamp))
            else:
                for xx in ck.chunk(2, -2):
                    x2 = xx.contiguous().view(nSamp, -1)
                    if self.pooling == 'max':
                        x3 = torch.max(x2, 1)
                        x_max.append(x3.values)
                    else:
                        x3 = torch.mean(x2, 1)
                        x_max.append(x3)
        x = torch.stack(x_max, 1)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nCls': 4}]
