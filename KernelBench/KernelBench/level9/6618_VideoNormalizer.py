import torch
import torch.nn as nn


class VideoNormalizer(nn.Module):

    def __init__(self):
        super(VideoNormalizer, self).__init__()
        self.scale = nn.Parameter(torch.Tensor([255.0]), requires_grad=False)
        self.mean = nn.Parameter(torch.Tensor([0.485, 0.456, 0.406]),
            requires_grad=False)
        self.std = nn.Parameter(torch.Tensor([0.229, 0.224, 0.225]),
            requires_grad=False)

    def forward(self, video):
        video = video.float()
        video = (video / self.scale - self.mean) / self.std
        return video.permute(0, 3, 1, 2)


def get_inputs():
    return [torch.rand([4, 4, 4, 3])]


def get_init_inputs():
    return [[], {}]
