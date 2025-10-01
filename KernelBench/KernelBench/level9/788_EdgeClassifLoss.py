import torch
import torch.optim


class EdgeClassifLoss(torch.nn.Module):

    def __init__(self, normalize=torch.nn.Sigmoid(), loss=torch.nn.BCELoss(
        reduction='mean')):
        super(EdgeClassifLoss, self).__init__()
        if isinstance(loss, torch.nn.BCELoss):
            self.loss = lambda preds, target: loss(preds, target)
        else:
            self.loss = loss
        self.normalize = normalize

    def forward(self, raw_scores, target):
        """
        outputs is the output of siamese network (bs,n_vertices,n_vertices)
        """
        preds = self.normalize(raw_scores)
        loss = self.loss(preds, target)
        return torch.mean(loss)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
