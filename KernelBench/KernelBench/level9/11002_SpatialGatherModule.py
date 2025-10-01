import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._C
import torch.serialization


class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale=1.0):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale
        assert self.scale > 0.0

    def forward(self, feats, prev_logits):
        """Forward function."""
        batch_size, num_classes, _height, _width = prev_logits.size()
        channels = feats.size(1)
        prev_logits = prev_logits.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        feats = feats.permute(0, 2, 1)
        probs = F.softmax(self.scale * prev_logits, dim=2)
        out_context = torch.matmul(probs, feats)
        out_context = out_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return out_context


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
