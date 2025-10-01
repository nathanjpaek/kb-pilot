from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn


class L1DepthLoss(nn.Module):
    """Custom L1 loss for depth sequences."""

    def __init__(self, args):
        super(L1DepthLoss, self).__init__()
        self.args = args
        self.word_dim = 1

    def forward(self, predictions, label_batch, length_batch):
        """ Computes L1 loss on depth sequences.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted depths
      label_batch: A pytorch batch of true depths
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
        total_sents = torch.sum(length_batch != 0).float()
        labels_1s = (label_batch != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_batch * labels_1s
        if total_sents > 0:
            loss_per_sent = torch.sum(torch.abs(predictions_masked -
                labels_masked), dim=self.word_dim)
            normalized_loss_per_sent = loss_per_sent / length_batch.float()
            batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
        else:
            batch_loss = torch.tensor(0.0, device=self.args['device'])
        return batch_loss, total_sents


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'args': _mock_config()}]
