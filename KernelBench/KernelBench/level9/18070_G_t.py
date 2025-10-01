from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class G_t(nn.Module):

    def __init__(self, args):
        super(G_t, self).__init__()
        self._relu = nn.ReLU()
        self._ws1 = nn.Linear(args.image_feature_dim, args.
            Vt_middle_feature_dim, bias=False)
        self._ws2 = nn.Linear(args.Vt_middle_feature_dim, args.
            video_feature_dim, bias=False)
        self._init_weights()

    def _init_weights(self, init_range=0.1):
        self._ws1.weight.data.uniform_(-init_range, init_range)
        self._ws2.weight.data.uniform_(-init_range, init_range)

    def forward(self, image_input):
        image_size = image_input.size()
        image_compressed_embeddings = image_input.view(-1, image_size[2])
        v_t = self._relu(self._ws1(image_compressed_embeddings))
        fake_video = self._relu(self._ws2(v_t)).view(image_size[0],
            image_size[1], -1)
        return fake_video


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config(image_feature_dim=4,
        Vt_middle_feature_dim=4, video_feature_dim=4)}]
