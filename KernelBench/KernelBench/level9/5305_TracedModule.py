import torch
import torch.quantization
import torch.onnx
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


class TracedModule(torch.nn.Module):

    def forward(self, x):
        x = x.type(torch.float32)
        return torch.floor(torch.sqrt(x) / 5.0)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
