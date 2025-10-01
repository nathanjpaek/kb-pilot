import torch
import torch.onnx


class StackTime(torch.nn.Module):
    __constants__ = ['factor']

    def __init__(self, factor):
        super().__init__()
        self.factor = int(factor)

    def forward(self, x, x_lens):
        seq = [x]
        for i in range(1, self.factor):
            tmp = torch.zeros_like(x)
            tmp[:-i, :, :] = x[i:, :, :]
            seq.append(tmp)
        x_lens = torch.ceil(x_lens.float() / self.factor).int()
        return torch.cat(seq, dim=2)[::self.factor, :, :], x_lens


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'factor': 4}]
