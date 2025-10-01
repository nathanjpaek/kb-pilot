import torch
from torch import nn


class SmoothContourLoss(nn.Module):
    """
        Loss function that contains smoothness loss derived from ss-with-RIM
        and contour-aware loss.
        Smoothness loss concerns about smoothness of local patterns, while
        contour-aware loss is interested in whether two patches are divided.
        Cross entropy (or KL divergence) is applied to calculate Contour-aware 
        loss. When calculating the gradients of probability, i.e. dp, and that 
        of image, i.e. di, we desire that the distribution of dp should appoximate
        that of di.

        Args:
            logits: torch.tensor
                A trainable tensor of shape (b, nSpixel, h, w)
                It should be softmaxed before calling this loss function.
            image: torch.tensor
                A tensor derived from color channels of input with shape
                (b, c, h, w)
            sigma: float
                Parameter of transformed Gaussian kernel size
            weights: List[float]
                A List contains 2 coefficients that control the amplitudes of
                2 losses
            thresh: float
                Parameter for controling the amplitude of edge
            margin: int
                Parameter for finding edge width
    """

    def __init__(self, sigma=2, weights=[1, 1], thresh=1.0, margin=1):
        super().__init__()
        self.sigma = 2 * sigma ** 2
        self.weights = weights
        self.thresh = thresh
        self.margin = margin

    def forward(self, logits, image):
        dp, di = self.get_gradients(logits, image)
        smooth = 0.0
        contour = 0.0
        for idx in range(len(dp)):
            smooth += self.smooth_loss(dp[idx], di[idx])
            contour += self.contour_loss(dp[idx], di[idx])
        return self.weights[0] * smooth + self.weights[1] * contour

    def get_gradients(self, logits, image):
        dp_dx = logits[..., :-self.margin] - logits[..., self.margin:]
        dp_dy = logits[..., :-self.margin, :] - logits[..., self.margin:, :]
        di_dx = image[..., :-self.margin] - image[..., self.margin:]
        di_dy = image[..., :-self.margin, :] - image[..., self.margin:, :]
        return [dp_dx, dp_dy], [di_dx, di_dy]

    def smooth_loss(self, dp, di):
        return (dp.abs().sum(1) * (-di.pow(2).sum(1) / self.sigma).exp()).mean(
            )

    def contour_loss(self, dp, di):
        di_norm = di.pow(2)
        di_min = di_norm.min(-1, keepdim=True).values.min(-2, keepdim=True
            ).values
        di_max = di_norm.max(-1, keepdim=True).values.max(-2, keepdim=True
            ).values
        di_norm = ((di_norm - di_min) / (di_max - di_min + 1e-16)).sum(1) * 2
        isValidEdges = di_norm > self.thresh
        dp_valid = dp.abs().sum(1) * isValidEdges
        di_valid = di_norm * isValidEdges
        return -(di_valid * torch.log(dp_valid + 1e-16)).mean()


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
