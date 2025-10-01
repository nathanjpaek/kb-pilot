import torch
import torch.nn as nn


class TverskyLoss(nn.Module):
    """Tversky Loss.

    .. seealso::
        Salehi, Seyed Sadegh Mohseni, Deniz Erdogmus, and Ali Gholipour. "Tversky loss function for image segmentation
        using 3D fully convolutional deep networks." International Workshop on Machine Learning in Medical Imaging.
        Springer, Cham, 2017.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Notes:
        - setting alpha=beta=0.5: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1706.05721.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def tversky_index(self, y_pred, y_true):
        """Compute Tversky index.

        Args:
            y_pred (torch Tensor): Prediction.
            y_true (torch Tensor): Target.

        Returns:
            float: Tversky index.
        """
        y_true = y_true.float()
        tp = torch.sum(y_true * y_pred)
        fn = torch.sum(y_true * (1 - y_pred))
        fp = torch.sum((1 - y_true) * y_pred)
        numerator = tp + self.smooth
        denominator = tp + self.alpha * fp + self.beta * fn + self.smooth
        tversky_label = numerator / denominator
        return tversky_label

    def forward(self, input, target):
        n_classes = input.shape[1]
        tversky_sum = 0.0
        for i_label in range(n_classes):
            y_pred, y_true = input[:, i_label], target[:, i_label]
            tversky_sum += self.tversky_index(y_pred, y_true)
        return -tversky_sum / n_classes


class FocalTverskyLoss(TverskyLoss):
    """Focal Tversky Loss.

    .. seealso::
        Abraham, Nabila, and Naimul Mefraz Khan. "A novel focal tversky loss function with improved attention u-net for
        lesion segmentation." 2019 IEEE 16th International Symposium on Biomedical Imaging (ISBI 2019). IEEE, 2019.

    Args:
        alpha (float): Weight of false positive voxels.
        beta  (float): Weight of false negative voxels.
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.
        smooth (float): Epsilon to avoid division by zero, when both Numerator and Denominator of Tversky are zeros.

    Attributes:
        gamma (float): Typically between 1 and 3. Control between easy background and hard ROI training examples.

    Notes:
        - setting alpha=beta=0.5 and gamma=1: Equivalent to DiceLoss.
        - default parameters were suggested by https://arxiv.org/pdf/1810.07842.pdf .
    """

    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1.0):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, input, target):
        n_classes = input.shape[1]
        focal_tversky_sum = 0.0
        for i_label in range(n_classes):
            y_pred, y_true = input[:, i_label], target[:, i_label]
            tversky_index = self.tversky.tversky_index(y_pred, y_true)
            focal_tversky_sum += torch.pow(1 - tversky_index, exponent=1 /
                self.gamma)
        return focal_tversky_sum / n_classes


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
