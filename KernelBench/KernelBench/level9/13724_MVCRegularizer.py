import torch
import torch.nn.parallel
import torch.utils.data


class MVCRegularizer(torch.nn.Module):
    """
    penalize MVC with large absolute value and negative values
    alpha * large_weight^2 + beta * (negative_weight)^2
    """

    def __init__(self, alpha=1.0, beta=1.0, threshold=5.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.threshold = threshold

    def forward(self, weights):
        loss = 0
        if self.alpha > 0:
            large_loss = torch.log(torch.nn.functional.relu(weights.abs() -
                self.threshold) + 1)
            loss += torch.mean(large_loss) * self.alpha
        if self.beta > 0:
            neg_loss = torch.nn.functional.relu(-weights)
            neg_loss = neg_loss ** 2
            loss += torch.mean(neg_loss) * self.beta
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
