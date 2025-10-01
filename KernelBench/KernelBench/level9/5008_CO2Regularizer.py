import torch


class MemoryBankModule(torch.nn.Module):
    """Memory bank implementation

    This is a parent class to all loss functions implemented by the lightly
    Python package. This way, any loss can be used with a memory bank if 
    desired.

    Attributes:
        size:
            Number of keys the memory bank can store. If set to 0,
            memory bank is not used.

    Examples:
        >>> class MyLossFunction(MemoryBankModule):
        >>>
        >>>     def __init__(self, memory_bank_size: int = 2 ** 16):
        >>>         super(MyLossFunction, self).__init__(memory_bank_size)
        >>>
        >>>     def forward(self, output: torch.Tensor,
        >>>                 labels: torch.Tensor = None):
        >>>
        >>>         output, negatives = super(
        >>>             MyLossFunction, self).forward(output)
        >>>
        >>>         if negatives is not None:
        >>>             # evaluate loss with negative samples
        >>>         else:
        >>>             # evaluate loss without negative samples

    """

    def __init__(self, size: 'int'=2 ** 16):
        super(MemoryBankModule, self).__init__()
        if size < 0:
            msg = f'Illegal memory bank size {size}, must be non-negative.'
            raise ValueError(msg)
        self.size = size
        self.bank = None
        self.bank_ptr = None

    @torch.no_grad()
    def _init_memory_bank(self, dim: 'int'):
        """Initialize the memory bank if it's empty

        Args:
            dim:
                The dimension of the which are stored in the bank.

        """
        self.bank = torch.randn(dim, self.size)
        self.bank = torch.nn.functional.normalize(self.bank, dim=0)
        self.bank_ptr = torch.LongTensor([0])

    @torch.no_grad()
    def _dequeue_and_enqueue(self, batch: 'torch.Tensor'):
        """Dequeue the oldest batch and add the latest one

        Args:
            batch:
                The latest batch of keys to add to the memory bank.

        """
        batch_size = batch.shape[0]
        ptr = int(self.bank_ptr)
        if ptr + batch_size >= self.size:
            self.bank[:, ptr:] = batch[:self.size - ptr].T.detach()
            self.bank_ptr[0] = 0
        else:
            self.bank[:, ptr:ptr + batch_size] = batch.T.detach()
            self.bank_ptr[0] = ptr + batch_size

    def forward(self, output: 'torch.Tensor', labels: 'torch.Tensor'=None,
        update: 'bool'=False):
        """Query memory bank for additional negative samples

        Args:
            output:
                The output of the model.
            labels:
                Should always be None, will be ignored.

        Returns:
            The output if the memory bank is of size 0, otherwise the output
            and the entries from the memory bank.

        """
        if self.size == 0:
            return output, None
        _, dim = output.shape
        if self.bank is None:
            self._init_memory_bank(dim)
        bank = self.bank.clone().detach()
        if update:
            self._dequeue_and_enqueue(output)
        return output, bank


class CO2Regularizer(MemoryBankModule):
    """Implementation of the CO2 regularizer [0] for self-supervised learning.

    [0] CO2, 2021, https://arxiv.org/abs/2010.02217

    Attributes:
        alpha:
            Weight of the regularization term.
        t_consistency:
            Temperature used during softmax calculations.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 to use the second batch for negative samples.

    Examples:
        >>> # initialize loss function for MoCo
        >>> loss_fn = NTXentLoss(memory_bank_size=4096)
        >>>
        >>> # initialize CO2 regularizer
        >>> co2 = CO2Regularizer(alpha=1.0, memory_bank_size=4096)
        >>>
        >>> # generate two random trasnforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through the MoCo model
        >>> out0, out1 = model(t0, t1)
        >>> 
        >>> # calculate loss and apply regularizer
        >>> loss = loss_fn(out0, out1) + co2(out0, out1)

    """

    def __init__(self, alpha: 'float'=1, t_consistency: 'float'=0.05,
        memory_bank_size: 'int'=0):
        super(CO2Regularizer, self).__init__(size=memory_bank_size)
        self.log_target = True
        try:
            self.kl_div = torch.nn.KLDivLoss(reduction='batchmean',
                log_target=True)
        except TypeError:
            self.log_target = False
            self.kl_div = torch.nn.KLDivLoss(reduction='batchmean')
        self.t_consistency = t_consistency
        self.alpha = alpha

    def _get_pseudo_labels(self, out0: 'torch.Tensor', out1: 'torch.Tensor',
        negatives: 'torch.Tensor'=None):
        """Computes the soft pseudo labels across negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images (query).
                Shape: bsz x n_ftrs
            out1:
                Output projections of the second set of transformed images (positive sample).
                Shape: bsz x n_ftrs
            negatives:
                Negative samples to compare against. If this is None, the second
                batch of images will be used as negative samples.
                Shape: memory_bank_size x n_ftrs

        Returns:
            Log probability that a positive samples will classify each negative
            sample as the positive sample.
            Shape: bsz x (bsz - 1) or bsz x memory_bank_size

        """
        batch_size, _ = out0.shape
        if negatives is None:
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, out1.t()])
            l_neg = l_neg.masked_select(~torch.eye(batch_size, dtype=bool,
                device=l_neg.device)).view(batch_size, batch_size - 1)
        else:
            negatives = negatives
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, negatives.clone().
                detach()])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits = logits / self.t_consistency
        return torch.nn.functional.log_softmax(logits, dim=-1)

    def forward(self, out0: 'torch.Tensor', out1: 'torch.Tensor'):
        """Computes the CO2 regularization term for two model outputs.

        Args:
            out0:
                Output projections of the first set of transformed images.
            out1:
                Output projections of the second set of transformed images.

        Returns:
            The regularization term multiplied by the weight factor alpha.

        """
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)
        out1, negatives = super(CO2Regularizer, self).forward(out1, update=True
            )
        p = self._get_pseudo_labels(out0, out1, negatives)
        q = self._get_pseudo_labels(out1, out0, negatives)
        if self.log_target:
            div = self.kl_div(p, q) + self.kl_div(q, p)
        else:
            div = self.kl_div(p, torch.exp(q)) + self.kl_div(q, torch.exp(p))
        return self.alpha * 0.5 * div


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
