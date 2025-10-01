import torch


class Unfold(torch.nn.Module):
    """Module for unfolding tensor.

    Performs strided crops on 2d (image) tensors. Stride is assumed to be half the crop size.

    """

    def __init__(self, img_size, fold_size):
        """

        Args:
            img_size: Input size.
            fold_size: Crop size.
        """
        super().__init__()
        fold_stride = fold_size // 2
        self.fold_size = fold_size
        self.fold_stride = fold_stride
        self.n_locs = 2 * (img_size // fold_size) - 1
        self.unfold = torch.nn.Unfold((self.fold_size, self.fold_size),
            stride=(self.fold_stride, self.fold_stride))

    def forward(self, x):
        """Unfolds tensor.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Unfolded tensor.

        """
        N = x.size(0)
        x = self.unfold(x).reshape(N, -1, self.fold_size, self.fold_size, 
            self.n_locs * self.n_locs).permute(0, 4, 1, 2, 3).reshape(N *
            self.n_locs * self.n_locs, -1, self.fold_size, self.fold_size)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'img_size': 4, 'fold_size': 4}]
