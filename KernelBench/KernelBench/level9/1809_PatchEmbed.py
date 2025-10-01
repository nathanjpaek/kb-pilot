import torch
import torch.nn.functional as F
import torch.nn as nn
import torch._C
import torch.serialization


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (tuple[int]): Patch token size. Default: (4, 4, 4).
        in_chans (int): Number of input image channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=(4, 4, 4), in_chans=1, embed_dim=96,
        norm_layer=None, use_spectral_aggregation='None'):
        super().__init__()
        self._patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
            stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None
        self.use_spectral_aggregation = use_spectral_aggregation
        if self.use_spectral_aggregation == 'token':
            self.spectral_aggregation_token = nn.Parameter(data=torch.empty
                (embed_dim), requires_grad=True)
            trunc_normal_(self.spectral_aggregation_token, std=0.02)

    def forward(self, x):
        """Forward function."""
        if self.use_spectral_aggregation != 'None':
            x = F.instance_norm(x)
            x = torch.unsqueeze(x, 1)
        _, _, S, H, W = x.size()
        if W % self._patch_size[2] != 0:
            x = F.pad(x, [0, self._patch_size[2] - W % self._patch_size[2]])
        if H % self._patch_size[1] != 0:
            x = F.pad(x, [0, 0, 0, self._patch_size[1] - H % self.
                _patch_size[1]])
        if S % self._patch_size[0] != 0:
            x = F.pad(x, [0, 0, 0, 0, 0, self._patch_size[0] - S % self.
                _patch_size[0]])
        x = self.proj(x)
        if self.use_spectral_aggregation == 'token':
            _b, _c, _s, _h, _w = x.shape
            token = self.spectral_aggregation_token.view(1, -1, 1, 1, 1
                ).repeat(_b, 1, 1, _h, _w)
            x = torch.cat((token, x), dim=2)
        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Ws, Wh, Ww)
        return x


def get_inputs():
    return [torch.rand([4, 1, 64, 64, 64])]


def get_init_inputs():
    return [[], {}]
