import torch
import torch.nn as nn


class BinaryCrossEntropyLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.

    Reference:
        Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.

    With label smoothing, the label :math:`y` for a class is computed by

    .. math::
        egin{equation}
        (1 - \\eps) 	imes y + rac{\\eps}{K},
        \\end{equation}

    where :math:`K` denotes the number of classes and :math:`\\eps` is a weight. When
    :math:`\\eps = 0`, the loss function reduces to the normal cross entropy.

    Args:
        eps (float, optional): weight. Default is 0.1.
        use_gpu (bool, optional): whether to use gpu devices. Default is True.
        label_smooth (bool, optional): whether to apply label smoothing. Default is True.
    """

    def __init__(self, eps=0.1, use_gpu=True, label_smooth=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.eps = eps if label_smooth else 0
        self.use_gpu = use_gpu
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): prediction matrix (before softmax) with
                shape (batch_size, num_classes).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
                Each position contains the label index.
        """
        inputs = torch.sigmoid(inputs)
        if self.use_gpu:
            targets = targets
        targets = (1 - self.eps) * targets + self.eps / 2
        return self.loss_fn(inputs, targets)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
