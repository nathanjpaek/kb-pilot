import logging
import torch
import torch.nn as nn


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get the root logger.

    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If `log_file` is specified, a FileHandler will
    also be added. The name of the root logger is the top-level package name,
    e.g., "mmedit".

    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.

    Returns:
        logging.Logger: The root logger.
    """
    logger = get_logger(__name__.split('.')[0], log_file, log_level)
    return logger


class SRCNN(nn.Module):
    """SRCNN network structure for image super resolution.

    SRCNN has three conv layers. For each layer, we can define the
    `in_channels`, `out_channels` and `kernel_size`.
    The input image will first be upsampled with a bicubic upsampler, and then
    super-resolved in the HR spatial size.

    Paper: Learning a Deep Convolutional Network for Image Super-Resolution.

    Args:
        channels (tuple[int]): A tuple of channel numbers for each layer
            including channels of input and output . Default: (3, 64, 32, 3).
        kernel_sizes (tuple[int]): A tuple of kernel sizes for each conv layer.
            Default: (9, 1, 5).
        upscale_factor (int): Upsampling factor. Default: 4.
    """

    def __init__(self, channels=(3, 64, 32, 3), kernel_sizes=(9, 1, 5),
        upscale_factor=4):
        super().__init__()
        assert len(channels
            ) == 4, f'The length of channel tuple should be 4, but got {len(channels)}'
        assert len(kernel_sizes
            ) == 3, f'The length of kernel tuple should be 3, but got {len(kernel_sizes)}'
        self.upscale_factor = upscale_factor
        self.img_upsampler = nn.Upsample(scale_factor=self.upscale_factor,
            mode='bicubic', align_corners=False)
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=
            kernel_sizes[0], padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=
            kernel_sizes[1], padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernel_size=
            kernel_sizes[2], padding=kernel_sizes[2] // 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        x = self.img_upsampler(x)
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        out = self.conv3(out)
        return out

    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is None:
            pass
        else:
            raise TypeError(
                f'"pretrained" must be a str or None. But received {type(pretrained)}.'
                )


def get_inputs():
    return [torch.rand([4, 3, 4, 4])]


def get_init_inputs():
    return [[], {}]
