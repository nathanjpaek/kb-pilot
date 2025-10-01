from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class Att(nn.Module):

    def __init__(self, args):
        super(Att, self).__init__()
        self._sigmoid = nn.Sigmoid()
        self._ws1 = nn.Linear(args.video_feature_dim, 1, bias=False)
        self._init_weights()

    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)

    def forward(self, video_input):
        video_size = video_input.size()
        image_compressed_embeddings = video_input.view(-1, video_size[2])
        attention = self._sigmoid(self._ws1(image_compressed_embeddings)).view(
            video_size[0], video_size[1], -1)
        attention = torch.transpose(attention, 1, 2).contiguous()
        return attention


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(video_feature_dim=4)}]
