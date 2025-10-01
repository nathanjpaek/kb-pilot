import torch
import torch.nn as nn


class ScaleHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = torch.flatten
        self.dot = torch.dot

    def forward(self, mag, height):
        curr_mag = self.flatten(mag, start_dim=1)
        curr_height = self.flatten(height, start_dim=1)
        batch_size = curr_mag.shape[0]
        length = curr_mag.shape[1]
        denom = torch.squeeze(torch.bmm(curr_height.view(batch_size, 1,
            length), curr_height.view(batch_size, length, 1))) + 0.01
        pinv = curr_height / denom.view(batch_size, 1)
        scale = torch.squeeze(torch.bmm(pinv.view(batch_size, 1, length),
            curr_mag.view(batch_size, length, 1)))
        return scale


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
