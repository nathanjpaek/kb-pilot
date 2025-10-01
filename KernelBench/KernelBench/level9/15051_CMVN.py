import torch
import torch.onnx


class CMVN(torch.nn.Module):
    eps = 1e-05

    @torch.no_grad()
    def forward(self, feat):
        mean = feat.mean(dim=2, keepdim=True)
        std = feat.std(dim=2, keepdim=True)
        feat = (feat - mean) / (std + CMVN.eps)
        return feat


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
