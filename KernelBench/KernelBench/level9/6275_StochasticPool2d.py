import torch
import torch.nn.functional as F


class StochasticPool2d(torch.nn.Module):

    def __init__(self, kernel_size=2, stride=2, padding=0):
        super(StochasticPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.grid_size = kernel_size
        self.padding = torch.nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x, s3pool_flag=False):
        if s3pool_flag or self.training:
            h, w = x.shape[-2:]
            n_h = h // self.grid_size
            n_w = w // self.grid_size
            n_h = int(n_h)
            n_w = int(n_w)
            x = self.padding(x)
            x = F.max_pool2d(x, self.kernel_size, 1)
            w_indices = []
            h_indices = []
            for i in range(n_w):
                position_offset = self.grid_size * i
                if i + 1 < n_w:
                    max_range = self.grid_size
                else:
                    max_range = w - position_offset
                if not self.training:
                    w_index = torch.LongTensor([0])
                else:
                    w_index = torch.LongTensor(1).random_(0, max_range)
                w_indices.append(torch.add(w_index, position_offset))
            for j in range(n_h):
                position_offset = self.grid_size * j
                if j + 1 < n_h:
                    max_range = self.grid_size
                else:
                    max_range = h - position_offset
                if not self.training:
                    h_index = torch.LongTensor([0])
                else:
                    h_index = torch.LongTensor(1).random_(0, max_range)
                h_indices.append(torch.add(h_index, position_offset))
            h_indices = torch.cat(h_indices, dim=0)
            w_indices = torch.cat(w_indices, dim=0)
            output = x[:, :, h_indices][:, :, :, w_indices]
            None
        else:
            output = F.avg_pool2d(x, self.kernel_size, self.stride)
        return output


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
