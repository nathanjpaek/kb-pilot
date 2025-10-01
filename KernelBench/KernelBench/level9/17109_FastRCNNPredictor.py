import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data
from torchvision.transforms import functional as F


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, n_fc_classif_layers=1,
        dropout=0.1, batched_nms=True):
        super(FastRCNNPredictor, self).__init__()
        self.n_fc_classif_layers = n_fc_classif_layers
        self.batched_nms = batched_nms
        self.fc_classif_layers = {i: nn.Linear(in_channels, in_channels) for
            i in range(n_fc_classif_layers - 1)}
        self.dropout = nn.Dropout(dropout)
        self.cls_score = nn.Linear(in_channels, num_classes)
        if self.batched_nms:
            self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        else:
            self.bbox_pred = nn.Linear(in_channels, 2 * 4)

    def forward(self, x, get_scores=True, get_deltas=True):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        if get_scores:
            scores = x + 0.0
            for lno, layer in self.fc_classif_layers.items():
                scores = F.relu(layer(scores))
                scores = self.dropout(scores)
            scores = self.cls_score(scores)
        else:
            scores = None
        bbox_deltas = self.bbox_pred(x) if get_deltas else None
        return scores, bbox_deltas


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'in_channels': 4, 'num_classes': 4}]
