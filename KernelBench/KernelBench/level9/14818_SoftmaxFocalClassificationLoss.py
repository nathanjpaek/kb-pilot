import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftmaxFocalClassificationLoss(nn.Module):
    """Criterion that computes Focal loss.
    According to [1], the Focal loss is computed as follows:
    .. math::
        \\text{FL}(p_t) = -\\alpha_t (1 - p_t)^{\\gamma} \\, \\text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Arguments:
        alpha (float): Weighting factor :math:`\\alpha \\in [0, 1]`.
        gamma (float): Focusing parameter :math:`\\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = kornia.losses.FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    References:
        [1] https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: 'float'=1.0, gamma: 'float'=2.0, reduction:
        'str'='none') ->None:
        super(SoftmaxFocalClassificationLoss, self).__init__()
        self.alpha: 'float' = alpha
        self.gamma: 'float' = gamma
        self.reduction: 'str' = reduction
        self.eps: 'float' = 1e-06

    def focal_loss(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor', alpha: 'float'=1.0, gamma: 'float'=2.0,
        reduction: 'str'='none', eps: 'float'=1e-08) ->torch.Tensor:
        """Function that computes Focal loss.
        See :class:`~kornia.losses.FocalLoss` for details.
        """
        if not torch.is_tensor(input):
            raise TypeError('Input type is not a torch.Tensor. Got {}'.
                format(type(input)))
        if not len(input.shape) >= 2:
            raise ValueError('Invalid input shape, we expect BxCx*. Got: {}'
                .format(input.shape))
        if input.size(0) != target.size(0):
            raise ValueError(
                'Expected input batch_size ({}) to match target batch_size ({}).'
                .format(input.size(0), target.size(0)))
        if not input.device == target.device:
            raise ValueError(
                'input and target must be in the same device. Got: {} and {}'
                .format(input.device, target.device))
        input_soft: 'torch.Tensor' = F.softmax(input, dim=1) + eps
        weight = torch.pow(-input_soft + 1.0, gamma)
        focal = -alpha * weight * torch.log(input_soft)
        loss_tmp = torch.sum(target * focal, dim=1, keepdims=True)
        if weights is None:
            return loss_tmp
        return loss_tmp * weights

    def forward(self, input: 'torch.Tensor', target: 'torch.Tensor',
        weights: 'torch.Tensor') ->torch.Tensor:
        return self.focal_loss(input, target, weights, self.alpha, self.
            gamma, self.reduction, self.eps)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
