import torch
import torch.utils
import torch
import torch.nn as nn


class NetworkExtension(nn.Module):

    def __init__(self, orig_num_classes, num_classes, auxiliary):
        super(NetworkExtension, self).__init__()
        self._auxiliary = auxiliary
        self.classifier = nn.Linear(orig_num_classes, num_classes)

    def forward(self, logits_logits_aux):
        logits = logits_logits_aux[0]
        logits_aux = logits_logits_aux[1]
        if self._auxiliary and self.training:
            logits_aux = torch.sigmoid(self.classifier(logits_aux))
        logits = torch.sigmoid(self.classifier(logits))
        return logits, logits_aux


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'orig_num_classes': 4, 'num_classes': 4, 'auxiliary': 4}]
