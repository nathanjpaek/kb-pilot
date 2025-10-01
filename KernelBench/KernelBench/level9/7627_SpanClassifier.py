import torch
import torch.nn as nn
from torch.nn import BCELoss


class SpanClassifier(nn.Module):
    """given the span embeddings, classify whether their relations"""

    def __init__(self, d_inp):
        super(SpanClassifier, self).__init__()
        self.d_inp = d_inp
        self.bilinear_layer = nn.Bilinear(d_inp, d_inp, 1)
        self.output = nn.Sigmoid()
        self.loss = BCELoss()

    def forward(self, span_emb_1, span_emb_2, label=None):
        """Calculate the similarity as bilinear product of span embeddings.

        Args:
            span_emb_1: [batch_size, hidden] (Tensor) hidden states for span_1
            span_emb_2: [batch_size, hidden] (Tensor) hidden states for span_2
            label: [batch_size] 0/1 Tensor, if none is supplied do prediction.
        """
        similarity = self.bilinear_layer(span_emb_1, span_emb_2)
        probs = self.output(similarity)
        outputs = similarity,
        if label is not None:
            cur_loss = self.loss(probs, label)
            outputs = (cur_loss,) + outputs
        return outputs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_inp': 4}]
