from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class G_u(nn.Module):

    def __init__(self, args):
        super(G_u, self).__init__()
        self._relu = nn.ReLU()
        self._ws1 = nn.Linear(args.video_feature_dim, args.
            Vu_middle_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.Vu_middle_feature_dim, args.
            image_feature_dim, bias=False)
        self._init_weights()

    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, video_input):
        video_size = video_input.size()
        video_compressed_embeddings = video_input.view(-1, video_size[2])
        v_u = self._relu(self._ws1(video_compressed_embeddings))
        fake_image = self._relu(self._ws2(v_u)).view(video_size[0],
            video_size[1], -1)
        return fake_image


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(video_feature_dim=4,
        Vu_middle_feature_dim=4, image_feature_dim=4)}]
