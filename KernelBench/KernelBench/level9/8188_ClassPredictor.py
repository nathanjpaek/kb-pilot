import torch
from torch import nn


class ClassPredictor(nn.Module):

    def __init__(self, nz_feat, max_object_classes):
        super(ClassPredictor, self).__init__()
        self.predictor = nn.Linear(nz_feat, max_object_classes)

    def forward(self, feats):
        class_logits = self.predictor(feats)
        return torch.nn.functional.log_softmax(class_logits)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nz_feat': 4, 'max_object_classes': 4}]
