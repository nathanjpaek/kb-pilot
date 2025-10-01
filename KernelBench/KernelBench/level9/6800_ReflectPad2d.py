import torch


class ReflectPad2d(torch.nn.Module):
    """ reflectionpad2d that can be transfered across onnx etc
        size : int (the size of padding)
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, ins):
        size = self.size
        l_list, r_list = [], []
        u_list, d_list = [], []
        for i in range(size):
            left = ins[:, :, :, size - i:size - i + 1]
            l_list.append(left)
            right = ins[:, :, :, i - size - 1:i - size]
            r_list.append(right)
        l_list.append(ins)
        ins = torch.cat(l_list + r_list[::-1], dim=3)
        for i in range(size):
            up = ins[:, :, size - i:size - i + 1, :]
            u_list.append(up)
            down = ins[:, :, i - size - 1:i - size, :]
            d_list.append(down)
        u_list.append(ins)
        ins = torch.cat(u_list + d_list[::-1], dim=2)
        return ins


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'size': 4}]
