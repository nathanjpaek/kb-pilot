import torch
import torch.nn as nn
import torch.nn.functional as F


def minimax_generator_loss(dgz, nonsaturating=True, reduction='mean'):
    if nonsaturating:
        target = torch.ones_like(dgz)
        return F.binary_cross_entropy_with_logits(dgz, target, reduction=
            reduction)
    else:
        target = torch.zeros_like(dgz)
        return -1.0 * F.binary_cross_entropy_with_logits(dgz, target,
            reduction=reduction)


class GeneratorLoss(nn.Module):
    """Base class for all generator losses.

    .. note:: All Losses meant to be minimized for optimizing the Generator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction='mean', override_train_ops=None):
        super(GeneratorLoss, self).__init__()
        self.reduction = reduction
        self.override_train_ops = override_train_ops
        self.arg_map = {}

    def set_arg_map(self, value):
        """Updates the ``arg_map`` for passing a different value to the ``train_ops``.

        Args:
            value (dict): A mapping of the ``argument name`` in the method signature and the
                variable name in the ``Trainer`` it corresponds to.

        .. note::
            If the ``train_ops`` signature is
            ``train_ops(self, gen, disc, optimizer_generator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_generator,
        device, batch_size, labels=None):
        """Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``generator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value = discriminator(fake)`
        3. :math:`loss = loss\\_function(value)`
        4. Backpropagate by computing :math:`\\nabla loss`
        5. Run a step of the optimizer for generator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_generator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``generator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(generator, discriminator,
                optimizer_generator, device, batch_size, labels)
        else:
            if labels is None and generator.label_type == 'required':
                raise Exception('GAN model requires labels for training')
            noise = torch.randn(batch_size, generator.encoding_dims, device
                =device)
            optimizer_generator.zero_grad()
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (
                    batch_size,), device=device)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            elif generator.label_type == 'generated':
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake)
            elif generator.label_type == 'generated':
                dgz = discriminator(fake, label_gen)
            else:
                dgz = discriminator(fake, labels)
            loss = self.forward(dgz)
            loss.backward()
            optimizer_generator.step()
            return loss.item()


class MinimaxGeneratorLoss(GeneratorLoss):
    """Minimax game generator loss from the original GAN paper `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(G) = log(1 - D(G(z)))

    The nonsaturating heuristic is also supported:

    .. math:: L(G) = -log(D(G(z)))

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`z` : A sample from the noise prior

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
        nonsaturating(bool, optional): Specifies whether to use the nonsaturating heuristic
            loss for the generator.
    """

    def __init__(self, reduction='mean', nonsaturating=True,
        override_train_ops=None):
        super(MinimaxGeneratorLoss, self).__init__(reduction,
            override_train_ops)
        self.nonsaturating = nonsaturating

    def forward(self, dgz):
        """Computes the loss for the given input.

        Args:
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \\*) where \\* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \\*).
        """
        return minimax_generator_loss(dgz, self.nonsaturating, self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
