import torch
import torch.utils.data
import torch.nn as nn


class MHSA(nn.Module):

    def __init__(self, n_dims, width=14, height=14, heads=4):
        super(MHSA, self).__init__()
        self.heads = heads
        self.query = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv2d(n_dims, n_dims, kernel_size=1)
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1,
            int(height)]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads,
            int(width), 1]), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        _c1, _c2, c3, _c4 = content_content.size()
        content_position = (self.rel_h + self.rel_w).view(1, self.heads, C //
            self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)
        content_position = (content_position if content_content.shape ==
            content_position.shape else content_position[:, :, :c3])
        assert content_content.shape == content_position.shape
        energy = content_content + content_position
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, width, height)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'n_dims': 4}]
