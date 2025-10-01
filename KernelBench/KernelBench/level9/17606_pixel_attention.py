import torch
from torch import nn


class pixel_attention(nn.Module):

    def __init__(self, in_channels, feature_size):
        super(pixel_attention, self).__init__()
        self.fc1 = nn.Linear(feature_size * feature_size, feature_size,
            bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(feature_size, feature_size * feature_size,
            bias=True)
        self.softmax = nn.Softmax()

    def forward(self, target_feature):
        b, c, h, w = target_feature.shape
        target_feature_resize = target_feature.view(b, c, h * w)
        p_f = torch.mean(target_feature_resize, dim=1)
        p_f = self.fc1(p_f)
        p_f = self.relu1(p_f)
        p_f = self.fc2(p_f)
        p_f = p_f.view(b, h * w)
        pixel_attention_weight = self.softmax(p_f)
        pixel_attention_weight = pixel_attention_weight.reshape(b, 1, h * w)
        return pixel_attention_weight


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'feature_size': 4}]
