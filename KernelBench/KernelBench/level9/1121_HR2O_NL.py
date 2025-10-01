import torch
import torch.nn as nn


class HR2O_NL(nn.Module):

    def __init__(self, hidden_dim=512, kernel_size=3, mlp_1x1=False):
        super(HR2O_NL, self).__init__()
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        self.conv_q = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
            padding=padding, bias=False)
        self.conv_k = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
            padding=padding, bias=False)
        self.conv_v = nn.Conv2d(hidden_dim, hidden_dim, kernel_size,
            padding=padding, bias=False)
        self.conv = nn.Conv2d(hidden_dim, hidden_dim, 1 if mlp_1x1 else
            kernel_size, padding=0 if mlp_1x1 else padding, bias=False)
        self.norm = nn.GroupNorm(1, hidden_dim, affine=True)
        self.dp = nn.Dropout(0.2)

    def forward(self, x):
        query = self.conv_q(x).unsqueeze(1)
        key = self.conv_k(x).unsqueeze(0)
        att = (query * key).sum(2) / self.hidden_dim ** 0.5
        att = nn.Softmax(dim=1)(att)
        value = self.conv_v(x)
        virt_feats = (att.unsqueeze(2) * value).sum(1)
        virt_feats = self.norm(virt_feats)
        virt_feats = nn.functional.relu(virt_feats)
        virt_feats = self.conv(virt_feats)
        virt_feats = self.dp(virt_feats)
        x = x + virt_feats
        return x


def get_inputs():
    return [torch.rand([4, 512, 64, 64])]


def get_init_inputs():
    return [[], {}]
