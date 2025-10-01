import logging
import torch


class ScalingFactor(torch.nn.Module):
    """
    Scale the output y of the layer s.t. the (mean) variance wrt. to the reference input x_ref is preserved.
    """

    def __init__(self):
        super().__init__()
        self.scale_factor = torch.nn.Parameter(torch.tensor(1.0),
            requires_grad=False)
        self.fitting_active = False

    def start_fitting(self):
        self.fitting_active = True
        self.variance_in = 0
        self.variance_out = 0
        self.num_samples = 0

    @torch.no_grad()
    def observe(self, x, x_ref=None):
        """
        Observe variances for output x and reference (input) x_ref.
        The scaling factor alpha is chosen s.t. Var(alpha * x) ~ Var(x_ref),
        or, if no x_ref is given, s.t. Var(alpha * x) ~ 1.
        """
        num_samples = x.shape[0]
        self.variance_out += torch.mean(torch.var(x, dim=0)) * num_samples
        if x_ref is None:
            self.variance_in += self.variance_out.new_tensor(num_samples)
        else:
            self.variance_in += torch.mean(torch.var(x_ref, dim=0)
                ) * num_samples
        self.num_samples += num_samples

    @torch.no_grad()
    def finalize_fitting(self):
        """
        Fit the scaling factor based on the observed variances.
        """
        if self.num_samples == 0:
            raise ValueError(
                'A ScalingFactor was not tracked. Add a forward call to track the variance.'
                )
        self.variance_in = self.variance_in / self.num_samples
        self.variance_out = self.variance_out / self.num_samples
        ratio = self.variance_out / self.variance_in
        value = torch.sqrt(1 / ratio)
        logging.info(
            f'Var_in: {self.variance_in.item():.3f}, Var_out: {self.variance_out.item():.3f}, Ratio: {ratio:.3f} => Scaling factor: {value:.3f}'
            )
        self.scale_factor.copy_(self.scale_factor * value)
        self.fitting_active = False

    def forward(self, x, x_ref=None):
        x = x * self.scale_factor
        if self.fitting_active:
            self.observe(x, x_ref)
        return x


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
