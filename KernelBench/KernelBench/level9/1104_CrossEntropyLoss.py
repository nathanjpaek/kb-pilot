import torch
import torch.nn as nn
import torch.utils.data.dataloader


class CrossEntropyLoss(nn.Module):
    """Custom cross-entropy loss"""

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.pytorch_ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1,
            reduction='sum')

    def forward(self, predictions, label_batch, length_batch):
        """
      Computes and returns CrossEntropyLoss.
      Ignores all entries where label_batch=-1
      Noralizes by the number of sentences in the batch.
      Args: 
        predictions: A pytorch batch of logits
        label_batch: A pytorch batch of label indices
        length_batch: A pytorch batch of sentence lengths
      Returns:
        A tuple of:
          cross_entropy_loss: average loss in the batch
          total_sents: number of sentences in the batch
      """
        batchlen, seqlen, class_count = predictions.size()
        total_sents = torch.sum(length_batch != 0).float()
        predictions = predictions.view(batchlen * seqlen, class_count)
        label_batch = label_batch.view(batchlen * seqlen).long()
        cross_entropy_loss = self.pytorch_ce_loss(predictions, label_batch
            ) / total_sents
        return cross_entropy_loss, total_sents


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([16]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
