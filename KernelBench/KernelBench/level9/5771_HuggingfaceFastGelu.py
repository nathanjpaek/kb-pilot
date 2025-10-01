import torch
import torch.nn
import torch.onnx
import torch.utils.checkpoint


class HuggingfaceFastGelu(torch.nn.Module):

    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 
            0.044715 * x * x)))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
