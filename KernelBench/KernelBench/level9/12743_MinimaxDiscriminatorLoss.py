import torch
import torch.nn as nn
import torch.nn.functional as F


def minimax_discriminator_loss(dx, dgz, label_smoothing=0.0, reduction='mean'):
    target_ones = torch.ones_like(dgz) * (1.0 - label_smoothing)
    target_zeros = torch.zeros_like(dx)
    loss = F.binary_cross_entropy_with_logits(dx, target_ones, reduction=
        reduction)
    loss += F.binary_cross_entropy_with_logits(dgz, target_zeros, reduction
        =reduction)
    return loss


class DiscriminatorLoss(nn.Module):
    """Base class for all discriminator losses.

    .. note:: All Losses meant to be minimized for optimizing the Discriminator must subclass this.

    Args:
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the outputs are averaged over batch size.
            If ``sum`` the elements of the output are summed.
        override_train_ops (function, optional): Function to be used in place of the default ``train_ops``
    """

    def __init__(self, reduction='mean', override_train_ops=None):
        super(DiscriminatorLoss, self).__init__()
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
            ``train_ops(self, gen, disc, optimizer_discriminator, device, batch_size, labels=None)``
            then we need to map ``gen`` to ``generator`` and ``disc`` to ``discriminator``.
            In this case we make the following function call
            ``loss.set_arg_map({"gen": "generator", "disc": "discriminator"})``.
        """
        self.arg_map.update(value)

    def train_ops(self, generator, discriminator, optimizer_discriminator,
        real_inputs, device, labels=None):
        """Defines the standard ``train_ops`` used by most losses. Losses which have a different
        training procedure can either ``subclass`` it **(recommended approach)** or make use of
        ``override_train_ops`` argument.

        The ``standard optimization algorithm`` for the ``discriminator`` defined in this train_ops
        is as follows:

        1. :math:`fake = generator(noise)`
        2. :math:`value_1 = discriminator(fake)`
        3. :math:`value_2 = discriminator(real)`
        4. :math:`loss = loss\\_function(value_1, value_2)`
        5. Backpropagate by computing :math:`\\nabla loss`
        6. Run a step of the optimizer for discriminator

        Args:
            generator (torchgan.models.Generator): The model to be optimized.
            discriminator (torchgan.models.Discriminator): The discriminator which judges the
                performance of the generator.
            optimizer_discriminator (torch.optim.Optimizer): Optimizer which updates the ``parameters``
                of the ``discriminator``.
            real_inputs (torch.Tensor): The real data to be fed to the ``discriminator``.
            device (torch.device): Device on which the ``generator`` and ``discriminator`` is present.
            batch_size (int): Batch Size of the data infered from the ``DataLoader`` by the ``Trainer``.
            labels (torch.Tensor, optional): Labels for the data.

        Returns:
            Scalar value of the loss.
        """
        if self.override_train_ops is not None:
            return self.override_train_ops(self, generator, discriminator,
                optimizer_discriminator, real_inputs, device, labels)
        else:
            if labels is None and (generator.label_type == 'required' or 
                discriminator.label_type == 'required'):
                raise Exception('GAN model requires labels for training')
            batch_size = real_inputs.size(0)
            noise = torch.randn(batch_size, generator.encoding_dims, device
                =device)
            if generator.label_type == 'generated':
                label_gen = torch.randint(0, generator.num_classes, (
                    batch_size,), device=device)
            optimizer_discriminator.zero_grad()
            if discriminator.label_type == 'none':
                dx = discriminator(real_inputs)
            elif discriminator.label_type == 'required':
                dx = discriminator(real_inputs, labels)
            else:
                dx = discriminator(real_inputs, label_gen)
            if generator.label_type == 'none':
                fake = generator(noise)
            elif generator.label_type == 'required':
                fake = generator(noise, labels)
            else:
                fake = generator(noise, label_gen)
            if discriminator.label_type == 'none':
                dgz = discriminator(fake.detach())
            elif generator.label_type == 'generated':
                dgz = discriminator(fake.detach(), label_gen)
            else:
                dgz = discriminator(fake.detach(), labels)
            loss = self.forward(dx, dgz)
            loss.backward()
            optimizer_discriminator.step()
            return loss.item()


class MinimaxDiscriminatorLoss(DiscriminatorLoss):
    """Minimax game discriminator loss from the original GAN paper `"Generative Adversarial Networks
    by Goodfellow et. al." <https://arxiv.org/abs/1406.2661>`_

    The loss can be described as:

    .. math:: L(D) = -[log(D(x)) + log(1 - D(G(z)))]

    where

    - :math:`G` : Generator
    - :math:`D` : Discriminator
    - :math:`x` : A sample from the data distribution
    - :math:`z` : A sample from the noise prior

    Args:
        label_smoothing (float, optional): The factor by which the labels (1 in this case) needs
            to be smoothened. For example, label_smoothing = 0.2 changes the value of the real
            labels to 0.8.
        reduction (str, optional): Specifies the reduction to apply to the output.
            If ``none`` no reduction will be applied. If ``mean`` the mean of the output.
            If ``sum`` the elements of the output will be summed.
        override_train_ops (function, optional): A function is passed to this argument,
            if the default ``train_ops`` is not to be used.
    """

    def __init__(self, label_smoothing=0.0, reduction='mean',
        override_train_ops=None):
        super(MinimaxDiscriminatorLoss, self).__init__(reduction,
            override_train_ops)
        self.label_smoothing = label_smoothing

    def forward(self, dx, dgz):
        """Computes the loss for the given input.

        Args:
            dx (torch.Tensor) : Output of the Discriminator with real data. It must have the
                                dimensions (N, \\*) where \\* means any number of additional
                                dimensions.
            dgz (torch.Tensor) : Output of the Discriminator with generated data. It must have the
                                 dimensions (N, \\*) where \\* means any number of additional
                                 dimensions.

        Returns:
            scalar if reduction is applied else Tensor with dimensions (N, \\*).
        """
        return minimax_discriminator_loss(dx, dgz, label_smoothing=self.
            label_smoothing, reduction=self.reduction)


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
