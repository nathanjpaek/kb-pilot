import logging
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List
import torch.onnx.operators
from functools import wraps


def singleton(cls):
    """
    Singleton decorator

    Args:
        cls: singleton class

    Returns:
        - an instance of a singleton class
    """
    instances = {}

    @wraps(cls)
    def getinstance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return getinstance


def get_invalid_class_mask(classes: 'int', invalid_classes: 'List'):
    """
    Create mask for invalid classes

    Args:
        classes: number of labels
        invalid_classes: invalid class list

    Returns:
        - mask for invalid class
            :math:`(1, L)` where L is the number of classes
    """
    invalid_class_mask = torch.zeros(classes).bool()
    if invalid_classes:
        for idx in invalid_classes:
            invalid_class_mask[idx] = True
    invalid_class_mask = invalid_class_mask.unsqueeze(dim=0)
    env = Environment()
    if env.device.startswith('cuda'):
        invalid_class_mask = invalid_class_mask
    return invalid_class_mask


@singleton
class Environment:
    """
    Environment is a running environment class.

    Args:
        profiling_window: profiling window size
        configs: configs for running tasks
        debug: running with debug information
        no_warning: do not output warning informations
        seed: initial seed for random and torch
        device: running device
        fp16: running with fp16
        no_progress_bar: do not show progress bar
        pb_interval: show progress bar with an interval
    """

    def __init__(self, configs=None, profiling_window: 'int'=0, debug:
        'bool'=False, no_warning: 'bool'=False, seed: 'int'=0, device:
        'str'=None, fp16: 'bool'=False, no_progress_bar: 'bool'=False,
        pb_interval: 'int'=1, custom_libs: 'str'=None):
        self.profiling_window = profiling_window
        self.configs = configs
        self.debug = debug
        self.no_warning = no_warning
        self.seed = seed
        self.fp16 = fp16
        self.no_progress_bar = no_progress_bar
        self.pb_interval = pb_interval
        self.distributed_world = 1
        self.rank = 0
        self.local_rank = 0
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        if self.device == 'cuda':
            self._init_cuda()
        self._init_log()
        self._init_seed()
        self._import_custom_lib(custom_libs)

    def _init_log(self):
        FORMAT = (
            f"%(asctime)s ｜ %(levelname)s | %(name)s |{f' RANK {self.rank} | ' if not self.is_master() else ' '}%(message)s"
            )
        logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d,%H:%M:%S',
            level=logging.INFO)
        if not self.is_master():
            logging.disable(logging.INFO)

    def _import_custom_lib(self, path):
        """
        Import library manually

        Args:
            path: external libraries split with `,`
        """
        if path:
            path = path.strip('\n')
            for line in path.split(','):
                logger.info(f'import module from {line}')
                line = line.replace('/', '.')
                importlib.import_module(line)

    def _init_cuda(self):
        """
        Initialize cuda device

        We assume that the user will not run ParaGen on more than one workers with only 1 GPU
        used on each worker.
        """
        if torch.cuda.device_count() > 1:
            hvd.init()
            torch.cuda.set_device(hvd.local_rank())
            self.rank = hvd.rank()
            self.local_rank = hvd.local_rank()
            self.distributed_world = hvd.size()
        torch.cuda.empty_cache()

    def _init_seed(self):
        """
        Initialize global seed
        """
        random.seed(self.seed)
        import torch
        torch.manual_seed(self.seed)
        if self.device == 'cuda':
            torch.manual_seed(self.seed)

    def is_master(self):
        """
        check the current process is the master process
        """
        return self.rank == 0


class LinearClassifier(nn.Module):
    """
    Classifier with only on a linear projection.

    Args:
        d_model: feature dimensionality
        labels: number of classes
        invalid_classes (List): class that is not allowed to produce
    """

    def __init__(self, d_model, labels, invalid_classes: 'List'=None):
        super().__init__()
        self._linear = nn.Linear(d_model, labels, bias=False)
        self._invalid_class_mask = get_invalid_class_mask(labels,
            invalid_classes) if invalid_classes else None

    def forward(self, x):
        """
        Args:
            x: feature to predict labels
                :math:`(*, D)`, where D is the feature dimension

        Returns:
            - log probability of each classes
                :math: `(*, L)`, where L is the number of classes
        """
        logits = self._linear(x)
        if self._invalid_class_mask is not None:
            logits = logits.masked_fill(self._invalid_class_mask, float('-inf')
                )
        logits = F.log_softmax(logits, dim=-1)
        return logits


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'labels': 4}]
