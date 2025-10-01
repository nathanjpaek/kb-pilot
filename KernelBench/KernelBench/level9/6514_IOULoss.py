import torch
import torch.nn as nn


def calculate_segmentation_statistics(outputs: 'torch.Tensor', targets:
    'torch.Tensor', class_dim: 'int'=1, threshold=None):
    """Compute calculate segmentation statistics.

    Args:
        outputs: torch.Tensor.
        targets: torch.Tensor.
        threshold: threshold for binarization of predictions.
        class_dim: indicates class dimension (K).

    Returns:
        True positives , false positives , false negatives for segmentation task.
    """
    num_dims = len(outputs.shape)
    assert num_dims > 2, 'Found only two dimensions, shape should be [bs , C , ...]'
    assert outputs.shape == targets.shape, 'shape mismatch'
    if threshold is not None:
        outputs = (outputs > threshold).float()
    dims = [dim for dim in range(num_dims) if dim != class_dim]
    true_positives = torch.sum(outputs * targets, dim=dims)
    false_positives = torch.sum(outputs * (1 - targets), dim=dims)
    false_negatives = torch.sum(targets * (1 - outputs), dim=dims)
    return true_positives, false_positives, false_negatives


class MetricMeter:
    """Base Class to structuring your metrics."""

    def accumulate(self, outputs, targets):
        """Method to accumulate outputs and targets per the batch."""
        raise NotImplementedError

    def compute(self):
        """Method to compute the metric on epoch end."""
        raise NotImplementedError

    def reset(self):
        """Method to reset the accumulation lists."""
        raise NotImplementedError


class IOU(MetricMeter):
    """Class which computes intersection over union."""

    def __init__(self, threshold: 'float'=None, class_dim: 'int'=1):
        """Constructor method for IOU.

        Args:
            threshold: threshold for binarization of predictions
            class_dim: indicates class dimension (K)

        Note:
            Supports only binary cases
        """
        self.threshold = threshold
        self.class_dim = class_dim
        self.eps = 1e-20
        self._outputs = []
        self._targets = []
        self.reset()

    def handle(self) ->str:
        """Method to get the class name.

        Returns:
            The class name
        """
        return self.__class__.__name__.lower()

    def accumulate(self, outputs: 'torch.Tensor', targets: 'torch.Tensor'):
        """Method to accumulate the outputs and targets.

        Args:
            outputs: [N, K, ...] tensor that for each of the N samples
                indicates the probability of the sample belonging to each of
                the K num_classes.
            targets:  binary [N, K, ...] tensor that encodes which of the K
                num_classes are associated with the N-th sample.
        """
        self._outputs.append(outputs)
        self._targets.append(targets)

    def compute(self) ->torch.Tensor:
        """Method to Compute IOU.

        Returns:
            The computed iou.
        """
        self._outputs = torch.cat(self._outputs)
        self._targets = torch.cat(self._targets)
        tp, fp, fn = calculate_segmentation_statistics(outputs=self.
            _outputs, targets=self._targets, threshold=self.threshold,
            class_dim=self.class_dim)
        union = tp + fp + fn
        score = (tp + self.eps * (union == 0).float()) / (tp + fp + fn +
            self.eps)
        return torch.mean(score)

    def reset(self):
        """Method to reset the accumulation lists."""
        self._outputs = []
        self._targets = []


class IOULoss(nn.Module):
    """Computes intersection over union Loss.

    IOULoss =  1 - iou_score
    """

    def __init__(self, class_dim=1):
        """Constructor method for IOULoss.

        Args:
            class_dim: indicates class dimension (K) for
                outputs and targets tensors (default = 1)
        """
        super(IOULoss, self).__init__()
        self.iou = IOU(threshold=None, class_dim=class_dim)

    def forward(self, outputs: 'torch.Tensor', targets: 'torch.Tensor'
        ) ->torch.Tensor:
        """Forward Method.

        Args:
            outputs: outputs from the net after applying activations.
            targets: The targets.

        Returns:
            The computed loss value.
        """
        self.iou.reset()
        self.iou.accumulate(outputs=outputs, targets=targets)
        return 1 - self.iou.compute()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
