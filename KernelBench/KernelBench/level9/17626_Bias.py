import torch
import torch.nn as nn


class Bias(nn.Module):

    def __init__(self):
        super(Bias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, feat_img, feat_sound):
        B, C, H, W = feat_sound.size()
        feat_img = feat_img.view(B, 1, C)
        z = torch.bmm(feat_img, feat_sound.view(B, C, H * W)).view(B, 1, H, W)
        z = z + self.bias
        return z

    def forward_nosum(self, feat_img, feat_sound):
        B, C, _H, _W = feat_sound.size()
        z = feat_img.view(B, C, 1, 1) * feat_sound
        z = z + self.bias
        return z

    def forward_pixelwise(self, feats_img, feat_sound):
        B, C, HI, WI = feats_img.size()
        B, C, HS, WS = feat_sound.size()
        feats_img = feats_img.view(B, C, HI * WI)
        feats_img = feats_img.transpose(1, 2)
        feat_sound = feat_sound.view(B, C, HS * WS)
        z = torch.bmm(feats_img, feat_sound).view(B, HI, WI, HS, WS)
        z = z + self.bias
        return z


def get_inputs():
    return [torch.rand([4, 1, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
