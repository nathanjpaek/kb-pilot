import torch


class SoftmaxAllocator(torch.nn.Module):
    """Portfolio creation by computing a softmax over the asset dimension with temperature.

    Parameters
    ----------
    temperature : None or float
        If None, then needs to be provided per sample during forward pass. If ``float`` then assumed
        to be always the same.

    formulation : str, {'analytical', 'variational'}
        Controls what way the problem is solved. If 'analytical' then using an explicit formula,
        however, one cannot decide on a `max_weight` different than 1. If `variational` then solved
        via convex optimization and one can set any `max_weight`.

    n_assets : None or int
        Only required and used if `formulation='variational`.

    max_weight : float
        A float between (0, 1] representing the maximum weight per asset.

    """

    def __init__(self, temperature=1, formulation='analytical', n_assets=
        None, max_weight=1):
        super().__init__()
        self.temperature = temperature
        if formulation not in {'analytical', 'variational'}:
            raise ValueError('Unrecognized formulation {}'.format(formulation))
        if formulation == 'variational' and n_assets is None:
            raise ValueError(
                'One needs to provide n_assets for the variational formulation.'
                )
        if formulation == 'analytical' and max_weight != 1:
            raise ValueError(
                'Cannot constraint weights via max_weight for analytical formulation'
                )
        if formulation == 'variational' and n_assets * max_weight < 1:
            raise ValueError(
                'One cannot create fully invested portfolio with the given max_weight'
                )
        self.formulation = formulation
        if formulation == 'analytical':
            self.layer = torch.nn.Softmax(dim=1)
        else:
            x = cp.Parameter(n_assets)
            w = cp.Variable(n_assets)
            obj = -x @ w - cp.sum(cp.entr(w))
            cons = [cp.sum(w) == 1.0, w <= max_weight]
            prob = cp.Problem(cp.Minimize(obj), cons)
            self.layer = CvxpyLayer(prob, [x], [w])

    def forward(self, x, temperature=None):
        """Perform forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        temperature : None or torch.Tensor
            If None, then using the `temperature` provided at construction time. Otherwise a `torch.Tensor` of shape
            `(n_samples,)` representing a per sample temperature.

        Returns
        -------
        weights : torch.Tensor
            Tensor of shape `(n_samples, n_assets`).

        """
        n_samples, _ = x.shape
        device, dtype = x.device, x.dtype
        if not (temperature is None) ^ (self.temperature is None):
            raise ValueError('Not clear which temperature to use')
        if temperature is not None:
            temperature_ = temperature
        else:
            temperature_ = float(self.temperature) * torch.ones(n_samples,
                dtype=dtype, device=device)
        inp = x / temperature_[..., None]
        return self.layer(inp
            ) if self.formulation == 'analytical' else self.layer(inp)[0]


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
