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


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like 
    the one described in the MoCo[1] paper.

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    [1] MoCo, 2020, https://arxiv.org/abs/1911.05722
    
    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank. 
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.

    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self, temperature: 'float'=0.5, memory_bank_size: 'int'=0):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction='mean')
        self.eps = 1e-08
        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'.format(
                self.temperature))

    def forward(self, out0: 'torch.Tensor', out1: 'torch.Tensor'):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as 
        negative samples.

            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Shape: (batch_size, embedding_size)
                out1:
                    Output projections of the second set of transformed images.
                    Shape: (batch_size, embedding_size)

            Returns:
                Contrastive Cross Entropy Loss value.

        """
        device = out0.device
        batch_size, _ = out0.shape
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)
        out1, negatives = super(NTXentLoss, self).forward(out1, update=out0
            .requires_grad)
        if negatives is not None:
            negatives = negatives
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)
            sim_neg = torch.einsum('nc,ck->nk', out0, negatives)
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=
                torch.long)
        else:
            output = torch.cat((out0, out1), axis=0)
            logits = torch.einsum('nc,mc->nm', output, output
                ) / self.temperature
            logits = logits[~torch.eye(2 * batch_size, dtype=torch.bool,
                device=out0.device)].view(2 * batch_size, -1)
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = torch.cat([labels + batch_size - 1, labels])
        loss = self.cross_entropy(logits, labels)
        return loss


def get_inputs():
    return [torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
