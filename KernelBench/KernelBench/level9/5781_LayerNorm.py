import torch
import torch.nn as nn
import torch.nn
import torch.onnx
import torch.utils.checkpoint


class LayerNorm(nn.Module):

    def __init__(self, hidden_size, epsilon, cast_fp16=True, formula=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=epsilon)
        self.layer_norm.bias.data.normal_(mean=0.0, std=0.1)
        self.layer_norm.weight.data.normal_(mean=0.0, std=0.5)
        self.cast_fp16 = cast_fp16
        self.formula = formula
        self.epsilon = epsilon

    @staticmethod
    def get_fused_op():
        return 'LayerNormalization'

    def my_layer_norm(self, x):
        if self.formula == 0:
            return self.layer_norm(x)
        x = x.float()
        u = x.mean(-1, keepdim=True)
        y = x - u
        s = y.pow(2).mean(-1, keepdim=True)
        z = y / torch.sqrt(s + self.epsilon)
        return self.layer_norm.weight.data * z + self.layer_norm.bias.data

    def forward(self, x):
        if self.cast_fp16 and x.dtype == torch.float16:
            y = self.my_layer_norm(x.to(torch.float32))
        else:
            y = self.my_layer_norm(x)
        return y,


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'epsilon': 4}]
