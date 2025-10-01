from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Capsule(nn.Module):

    def __init__(self, cfg):
        super(Capsule, self).__init__()
        self.input_dim_capsule = cfg.input_dim_capsule
        self.dim_capsule = cfg.dim_capsule
        self.num_capsule = cfg.num_capsule
        self.batch_size = cfg.batch_size
        self.share_weights = cfg.share_weights
        self.num_iterations = cfg.num_iterations
        if self.share_weights:
            W = torch.zeros(1, self.input_dim_capsule, self.num_capsule *
                self.dim_capsule)
        else:
            W = torch.zeros(self.batch_size, self.input_dim_capsule, self.
                num_capsule * self.dim_capsule)
        W = nn.init.xavier_normal_(W)
        self.W = nn.Parameter(W)

    def forward(self, x):
        """
        x: [B, L, H]      # 从 CNN / RNN 得到的结果
            L 作为 input_num_capsules, H 作为 input_dim_capsule
        """
        B, I, _ = x.size()
        O, F = self.num_capsule, self.dim_capsule
        u = torch.matmul(x, self.W)
        u = u.view(B, I, O, F).transpose(1, 2)
        b = torch.zeros_like(u[:, :, :, 0])
        for i in range(self.num_iterations):
            c = torch.softmax(b, dim=1)
            v = torch.einsum('boi,boif->bof', [c, u])
            v = self.squash(v)
            b = torch.einsum('bof,boif->boi', [v, u])
        return v

    @staticmethod
    def squash(x: 'torch.Tensor'):
        x_norm = x.norm(p=2, dim=-1, keepdim=True)
        mag = x_norm ** 2
        out = x / x_norm * mag / (1 + mag)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'cfg': _mock_config(input_dim_capsule=4, dim_capsule=4,
        num_capsule=4, batch_size=4, share_weights=4, num_iterations=4)}]
