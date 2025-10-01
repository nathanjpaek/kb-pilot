import torch
from torch.nn import functional as F
from torch import nn
from torchvision import models as models
import torch.onnx
import torch.nn


class LengthPredictionLoss(nn.Module):

    def __init__(self, max_delta=50):
        super().__init__()
        self.max_delta = max_delta

    def forward(self, logits, src_mask, tgt_mask):
        src_lens, tgt_lens = src_mask.sum(1), tgt_mask.sum(1)
        delta = (tgt_lens - src_lens + self.max_delta).clamp(0, self.
            max_delta * 2 - 1).long()
        loss = F.cross_entropy(logits, delta, reduction='mean')
        return {'length_prediction_loss': loss}


class LengthPredictor(nn.Module):

    def __init__(self, hidden_size, max_delta=50):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_delta = max_delta
        self._init_modules()
        self._init_loss()

    def forward(self, src, src_mask, tgt_len=None):
        src_mean = self._compute_mean_emb(src, src_mask)
        logits, delta = self._predict_delta(src_mean)
        return logits, delta

    def _predict_delta(self, src):
        logits = self.length_predictor(src)
        delta = logits.argmax(-1) - float(self.max_delta)
        return logits, delta

    def _compute_mean_emb(self, src, src_mask):
        mean_emb = (src * src_mask[:, :, None]).sum(1) / src_mask.sum(1)[:,
            None]
        return mean_emb

    def _init_modules(self):
        self.length_predictor = nn.Linear(self.hidden_size, self.max_delta * 2)

    def _init_loss(self):
        self.loss = LengthPredictionLoss(self.max_delta)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4}]
