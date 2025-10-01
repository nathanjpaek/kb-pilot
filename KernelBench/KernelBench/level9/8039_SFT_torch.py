import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import *


class SFT_torch(nn.Module):

    def __init__(self, sigma=0.1, *args, **kwargs):
        super(SFT_torch, self).__init__(*args, **kwargs)
        self.sigma = sigma

    def forward(self, emb_org):
        emb_org_norm = torch.norm(emb_org, 2, 1, True).clamp(min=1e-12)
        emb_org_norm = torch.div(emb_org, emb_org_norm)
        W = torch.mm(emb_org_norm, emb_org_norm.t())
        W = torch.div(W, self.sigma)
        T = F.softmax(W, 1)
        emb_sft = torch.mm(T, emb_org)
        return emb_sft


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
