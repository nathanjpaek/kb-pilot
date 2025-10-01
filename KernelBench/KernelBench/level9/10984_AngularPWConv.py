import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C
import torch.serialization


def normalize(x, dim, p=2, eps=1e-12):
    if torch.onnx.is_in_onnx_export():
        return OnnxLpNormalization.apply(x, dim, p, eps)
    else:
        return F.normalize(x, dim=dim, p=p, eps=eps)


class OnnxLpNormalization(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, axis=0, p=2, eps=1e-12):
        denom = x.norm(2, axis, True).clamp_min(eps).expand_as(x)
        return x / denom

    @staticmethod
    def symbolic(g, x, axis=0, p=2, eps=1e-12):
        return g.op('LpNormalization', x, axis_i=int(axis), p_i=int(p))


class AngularPWConv(nn.Module):

    def __init__(self, in_features, out_features, clip_output=False):
        super(AngularPWConv, self).__init__()
        self.in_features = in_features
        assert in_features > 0
        self.out_features = out_features
        assert out_features >= 2
        self.clip_output = clip_output
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight.data.normal_().renorm_(2, 1, 1e-05).mul_(100000.0)

    def forward(self, x):
        weight = normalize(self.weight, dim=1, p=2).view(self.out_features,
            self.in_features, 1, 1)
        out = F.conv2d(x, weight)
        if self.clip_output and not torch.onnx.is_in_onnx_export():
            out = out.clamp(-1.0, 1.0)
        return out


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_features': 4, 'out_features': 4}]
