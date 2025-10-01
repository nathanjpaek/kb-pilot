import torch
import torch.nn as nn


class BucketingEmbedding(nn.Module):

    def __init__(self, min_val, max_val, count, dim, use_log_scale=False):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.count = count
        self.dim = dim
        self.use_log_scale = use_log_scale
        if self.use_log_scale:
            self.min_val = torch.log2(torch.Tensor([self.min_val])).item()
            self.max_val = torch.log2(torch.Tensor([self.max_val])).item()
        self.main = nn.Embedding(count, dim)

    def forward(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return self.main(x)

    def get_class(self, x):
        """
        x - (bs, ) values
        """
        if self.use_log_scale:
            x = torch.log2(x)
        x = self.count * (x - self.min_val) / (self.max_val - self.min_val)
        x = torch.clamp(x, 0, self.count - 1).long()
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'min_val': 4, 'max_val': 4, 'count': 4, 'dim': 4}]
