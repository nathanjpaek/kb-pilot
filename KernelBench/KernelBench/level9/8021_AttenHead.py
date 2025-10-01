import math
import torch
from torch.nn import functional as F
from torch import nn


class AttenHead(nn.Module):

    def __init__(self, fdim, num_heads=1):
        super().__init__()
        self.num_heads = num_heads
        self.fatt = fdim // num_heads
        for i in range(num_heads):
            setattr(self, f'embd{i}', nn.Linear(fdim, self.fatt))
        for i in range(num_heads):
            setattr(self, f'fc{i}', nn.Linear(2 * self.fatt, self.fatt))
        self.fc = nn.Linear(self.fatt * num_heads, fdim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, fx_in, fp_in):
        fp_in = fp_in.squeeze(0)
        d = math.sqrt(self.fatt)
        Nx = len(fx_in)
        f = torch.cat([fx_in, fp_in])
        f = torch.stack([getattr(self, f'embd{i}')(f) for i in range(self.
            num_heads)])
        fx, fp = f[:, :Nx], f[:, Nx:]
        w = self.dropout(F.softmax(torch.matmul(fx, torch.transpose(fp, 1, 
            2)) / d, dim=2))
        fa = torch.cat([torch.matmul(w, fp), fx], dim=2)
        fa = torch.stack([F.relu(getattr(self, f'fc{i}')(fa[i])) for i in
            range(self.num_heads)])
        fa = torch.transpose(fa, 0, 1).reshape(Nx, -1)
        fx = F.relu(fx_in + self.fc(fa))
        w = torch.transpose(w, 0, 1)
        return fx, w


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'fdim': 4}]
