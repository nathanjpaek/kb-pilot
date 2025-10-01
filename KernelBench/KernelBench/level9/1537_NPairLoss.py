import torch


class NPairLoss(torch.nn.Module):

    def __init__(self, l2=0.05):
        """
        Basic N-Pair Loss as proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'
        Args:
            l2: float, weighting parameter for weight penality due to embeddings not being normalized.
        Returns:
            Nothing!
        """
        super(NPairLoss, self).__init__()
        self.l2 = l2

    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.
        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1 + torch.sum(torch.exp(anchor.reshape(1, -1).mm((
            negatives - positive).transpose(0, 1)))))

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.
        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor ** 2 + positive ** 2)

    def forward(self, batch):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
        Returns:
            n-pair loss (torch.Tensor(), batch-averaged)
        """
        loss = torch.stack([self.npair_distance(npair[0], npair[1], npair[2
            :]) for npair in batch])
        loss = loss + self.l2 * torch.mean(torch.stack([self.weightsum(
            npair[0], npair[1]) for npair in batch]))
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {}]
