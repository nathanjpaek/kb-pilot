from torch.nn import Module
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.nn.functional
import torch.autograd


class MarginLoss(Module):
    '\n    ## Margin loss for class existence\n\n    A separate margin loss is used for each output capsule and the total loss is the sum of them.\n    The length of each output capsule is the probability that class is present in the input.\n\n    Loss for each output capsule or class $k$ is,\n    $$\\mathcal{L}_k = T_k \\max(0, m^{+} - \\lVert\\mathbf{v}_k\rVert)^2 +\n    \\lambda (1 - T_k) \\max(0, \\lVert\\mathbf{v}_k\rVert - m^{-})^2$$\n\n    $T_k$ is $1$ if the class $k$ is present and $0$ otherwise.\n    The first component of the loss is $0$ when the class is not present,\n    and the second component is $0$ if the class is present.\n    The $\\max(0, x)$ is used to avoid predictions going to extremes.\n    $m^{+}$ is set to be $0.9$ and $m^{-}$ to be $0.1$ in the paper.\n\n    The $\\lambda$ down-weighting is used to stop the length of all capsules from\n    falling during the initial phase of training.\n    '

    def __init__(self, *, n_labels: int, lambda_: float=0.5, m_positive:
        float=0.9, m_negative: float=0.1):
        super().__init__()
        self.m_negative = m_negative
        self.m_positive = m_positive
        self.lambda_ = lambda_
        self.n_labels = n_labels

    def forward(self, v: 'torch.Tensor', labels: 'torch.Tensor'):
        """
        `v`, $\\mathbf{v}_j$ are the squashed output capsules.
        This has shape `[batch_size, n_labels, n_features]`; that is, there is a capsule for each label.

        `labels` are the labels, and has shape `[batch_size]`.
        """
        v_norm = torch.sqrt((v ** 2).sum(dim=-1))
        labels = torch.eye(self.n_labels, device=labels.device)[labels]
        loss = labels * F.relu(self.m_positive - v_norm) + self.lambda_ * (
            1.0 - labels) * F.relu(v_norm - self.m_negative)
        return loss.sum(dim=-1).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'n_labels': 4}]
